"""Verified task reuse and the paired Spartacus experiment runner."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import perf_counter
import unicodedata
from unittest.mock import patch
from urllib import request

from content import ContentReuse, THRESHOLD as CONTENT_THRESHOLD
from minhash import LSH, MinHash, SEED, shingles, similarity, words

MODEL = "gemini-3.5-flash-lite"
DATASETS = ("tickets", "bugs", "docs", "citations", "hosts", "configs", "revisions", "logs")
CORPUS = {"hosts", "configs", "code", "threads", "revisions", "logs"}
MODES = {"baseline": (False, False), "task": (True, False),
         "content": (False, True), "both": (True, True)}
TASK_THRESHOLD = 0.55
PROMPT = "Read task.json and INSTRUCTIONS.md. Solve the task and write the required JSON to answer.json."
ROOT = Path(__file__).resolve().parent


def text_key(value):
    return " ".join(words(value)) if isinstance(value, str) else ""


def code_key(value):
    return "".join(words(value)) if isinstance(value, str) else ""


def story_key(body):
    # Only case and whitespace are cosmetic here. Preserve numbers, punctuation and word order.
    return " ".join(unicodedata.normalize("NFC", body).casefold().split())


def identity(dataset, record, aliases):
    try:
        if dataset in CORPUS:
            scenario, kind = code_key(record["scenario"]), text_key(record["kind"])
            target = record.get("target")
            if target is not None and not isinstance(target, str):
                return None
            return (scenario, kind, code_key(target)) if scenario and kind else None
        if dataset == "tickets":
            fields = (text_key(record["product"]), code_key(record["error_code"]),
                      text_key(record["platform"]), story_key(record["body"]))
            return fields if all(fields) else None
        if dataset == "bugs":
            exception = record["exception"].split(".")[-1].casefold()
            frame = re.sub(r":\d+$", "", record["top_frame"].strip().casefold())
            frame = frame.replace(".py::", ".").replace("::", ".").replace("/", ".")
            series = re.search(r"\d+\.\d+", record["version"])
            fields = (exception, frame, series.group() if series else "", story_key(record["body"]))
            return fields if all(fields) else None
        if dataset == "docs":
            return document_identity(record)
        if dataset == "citations":
            return citation_identity(record["citation"], aliases)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


DOC_ID = re.compile(r"\b(?:CHG|INC|POL)[- ]?\d{3,4}\b|\bREL[- ]?\d+\.\d+\b", re.I)
DOC_STATUS = {
    "chg": (r"request(?:ed)?|pending", r"approved|authori[sz]ed", r"rejected|declined"),
    "inc": (r"ongoing|under investigation", r"resolved|closed"),
    "pol": (r"draft|for comment", r"effective|in force"),
}


def document_identity(record):
    identifiers = DOC_ID.findall(record["title"])
    if len(identifiers) != 1:
        return None
    identifier = identifiers[0].casefold().replace("-", "").replace(" ", "")
    family = identifier[:3]
    if family == "rel":
        return identifier, 0
    body = " ".join(words(record["body"]))
    classes = [
        index for index, pattern in enumerate(DOC_STATUS[family])
        if re.search(rf"\b(?:{pattern})\b", body)
    ]
    if len(classes) == 1 and not re.search(r"\b(?:not|never)\b", body):
        return identifier, classes[0]
    return None


def author_key(author):
    if "," in author:
        surname, given = author.split(",", 1)
    else:
        parts = author.split()
        surname, given = parts[-1], " ".join(parts[:-1])
    return text_key(surname), text_key(given)[:1]


def citation_identity(citation, aliases):
    doi = (citation.get("doi") or "").strip().casefold()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/|^doi:\s*", "", doi).rstrip(". ")
    title, venue = text_key(citation["title"]), text_key(citation["venue"])
    authors = citation["authors"]
    if not title or not venue or not isinstance(authors, list) or not authors:
        return None
    if any(not isinstance(author, str) or not author.strip() for author in authors):
        return None
    return doi, title, tuple(map(author_key, authors)), aliases.get(venue, venue), int(citation["year"])


def task_text(dataset, record):
    fields = ("subject", "title", "body", "stacktrace", "raw")
    return "\n".join(record[key] for key in fields if isinstance(record.get(key), str))


def history_answer(dataset, record):
    if dataset in CORPUS:
        return record.get("answer")
    if dataset == "citations":
        return record.get("verdict")
    if dataset == "docs":
        return {"category": record["category"], "duplicate_of": record["hid"]}
    keys = ("resolution_id",) if dataset == "tickets" else ("component", "root_cause")
    return {key: record[key] for key in keys}


@dataclass(frozen=True)
class Entry:
    source: str
    identity: tuple
    answer: dict


class TaskReuse:
    def __init__(self, dataset, history, *, aliases=None, threshold=TASK_THRESHOLD, hasher=None):
        self.dataset = dataset
        self.aliases = aliases or {}
        self.threshold = threshold
        self.hasher = hasher if hasher is not None else MinHash()
        self.index = LSH()
        self.entries = []
        self.answers = defaultdict(set)
        for row in history:
            self._add(row, history_answer(dataset, row), row["hid"])

    def _add(self, record, answer, source):
        key = identity(self.dataset, record, self.aliases)
        if key is None or not isinstance(answer, dict) or not answer:
            return
        signature = self.hasher.signature(shingles(task_text(self.dataset, record)))
        self.index.add(len(self.entries), signature)
        self.entries.append(Entry(source, key, deepcopy(answer)))
        self.answers[key].add(json.dumps(answer, sort_keys=True))

    def add_verified(self, record, answer, *, passed):
        if passed:
            self._add(record, answer, record["tid"])

    def find(self, record):
        key = identity(self.dataset, record, self.aliases)
        signature = self.hasher.signature(shingles(task_text(self.dataset, record)))
        if key is None or len(self.answers.get(key, ())) != 1:
            return None
        matches = [
            (score, index) for index, score in self.index.query(signature)
            if score >= self.threshold and self.entries[index].identity == key
        ]
        if not matches:
            return None
        score, index = max(matches, key=lambda item: (item[0], -item[1]))
        entry = self.entries[index]
        return deepcopy(entry.answer), entry.source, score


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def repair_checkpoint(path):
    """Keep completed rows if a process stopped during its last append."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    end_of_rows = data.rfind(b"\n") + 1
    try:
        json.loads(data[end_of_rows:])
    except (ValueError, UnicodeError):
        with path.open("r+b") as checkpoint:
            checkpoint.truncate(end_of_rows)
    else:
        with path.open("ab") as checkpoint:
            checkpoint.write(b"\n")


@dataclass
class Dataset:
    name: str
    root: Path

    @property
    def env(self):
        return self.root / "env"

    def load_history(self):
        return json_lines(self.env / "history.jsonl")

    def aliases(self):
        if self.name != "citations":
            return {}
        return {
            text_key(alias): text_key(venue["name"])
            for venue in read_json(self.env / "db.json")["venues"]
            for alias in [venue["name"], *venue["abbrev"]]
        }

    def read_tasks(self, limit, seed):
        tasks = [read_json(path) for path in sorted((self.env / "incoming").glob("*.json"))]
        if limit and limit < len(tasks):
            tasks = Random(f"{seed}:{self.name}").sample(tasks, limit)
        return sorted(tasks, key=lambda task: (task.get("seq", 0), task["tid"]))

    def stage(self, task, directory):
        for item in self.env.iterdir():
            if item.is_file() and not item.is_symlink():
                shutil.copyfile(item, directory / item.name)
        if self.name in CORPUS:
            matching = [
                path for path in (self.env / "scenarios").iterdir()
                if code_key(path.name) == code_key(task["scenario"])
            ]
            if len(matching) != 1:
                raise ValueError(f"cannot stage scenario for {task['tid']}")
            shutil.copytree(matching[0], directory, dirs_exist_ok=True, symlinks=True)
        (directory / "task.json").write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")

    def verify(self, tid, path):
        result = subprocess.run(
            [sys.executable, str(self.root / "verify.py"), "--tid", tid, "--answer", str(path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode not in (0, 1) or result.stderr:
            raise RuntimeError(f"Verifier process failed for {self.name}/{tid}")
        return result.returncode == 0


class FixturePolicy:
    """Allow Spartacus to read the staged fixture and write the answer."""

    def check(self, call):
        name = call["name"]
        if name in {"read_file", "grep", "list_files"}:
            return None
        if name in {"write_file", "edit_file"} and call["args"].get("path") in {"answer.json", "./answer.json"}:
            return None
        return "This benchmark allows writing only answer.json."


class Pace:
    def __init__(self, rpm, aborted):
        self.interval = 60 / rpm if rpm else 0
        self.lock = Lock()
        self.next_request = 0.0
        self.aborted = aborted

    def wait(self):
        with self.lock:
            delay = max(0, self.next_request - perf_counter())
            self.next_request = perf_counter() + delay + self.interval
        if self.aborted.wait(delay):
            raise InterruptedError("benchmark stopped")
        return delay


@dataclass
class Telemetry:
    content: ContentReuse | None
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    rate_wait_s: float = 0


ACTIVE = ContextVar("active_task", default=None)


def measured_provider(complete, pace):
    def run(model, system, messages, tools=None):
        stats = ACTIVE.get()
        if stats is None:
            return complete(model, system, messages, tools)
        stats.rate_wait_s += pace.wait()
        if stats.content is not None:
            messages = stats.content.render(messages)
        stats.requests += 1
        reply = complete(model, system, messages, tools)
        stats.input_tokens += reply["usage"]["input"]
        stats.output_tokens += reply["usage"]["output"]
        return reply
    return run


def run_agent(directory, stats, max_turns):
    from spartacus import Harness

    def on_event(kind, payload):
        if kind == "tool_start":
            stats.tool_calls += 1
        if kind == "tool_end" and stats.content is not None:
            stats.content.observe(payload["call"], harness.messages[-1], harness.messages)

    harness = Harness(
        workdir=directory, model=MODEL, policy=FixturePolicy(), on_event=on_event,
        max_turns=max_turns, enable_subagents=False, persist=False,
    )
    # Apply the same restricted tool surface for every configuration.
    harness.tools.pop("bash")
    harness.tools.pop("remember")
    token = ACTIVE.set(stats)
    try:
        harness.run(PROMPT)
    finally:
        ACTIVE.reset(token)


def trial(dataset, task, mode, task_reuse, args, initialization):
    task_enabled, content_enabled = MODES[mode]
    started = perf_counter()
    measured = Telemetry(ContentReuse(threshold=args.content_threshold) if content_enabled else None)
    match = task_reuse.find(task) if task_enabled else None
    lookup_s = perf_counter() - started if task_enabled else 0
    insertion_s = 0
    answer = None
    with TemporaryDirectory(prefix="assignment2-task-") as tmp:
        directory = Path(tmp)
        path = directory / "answer.json"
        if match:
            answer = match[0]
            path.write_text(json.dumps(answer), encoding="utf-8")
        else:
            dataset.stage(task, directory)
            run_agent(directory, measured, args.max_turns)
            try:
                answer = read_json(path)
            except (FileNotFoundError, ValueError):
                pass
        passed = dataset.verify(task["tid"], path)
        if task_enabled and not match and passed:
            before = perf_counter()
            task_reuse.add_verified(task, answer, passed=passed)
            insertion_s = perf_counter() - before
    content = measured.content
    return {
        "dataset": dataset.name, "tid": task["tid"], "mode": mode, "passed": passed,
        "tool_calls": measured.tool_calls, "model_requests": measured.requests,
        "input_tokens": measured.input_tokens, "output_tokens": measured.output_tokens,
        "wall_s": perf_counter() - started + initialization, "rate_wait_s": measured.rate_wait_s,
        "task_lookup_s": lookup_s, "task_index_s": initialization + insertion_s,
        "content_s": content.seconds if content else 0,
        "reuse_from": match[1] if match else None, "score": match[2] if match else None,
        "folds": content.hits if content else 0,
        "bytes_saved": content.bytes_saved if content else 0,
        "verified_answer": answer if task_enabled and not match and passed else None,
    }


def benchmark_dataset(dataset, tasks, args, aborted):
    path = args.out / f"{dataset.name}.jsonl"
    repair_checkpoint(path)
    previous = {(row["tid"], row["mode"]): row for row in json_lines(path)}
    history, aliases = dataset.load_history(), dataset.aliases()
    indexes = {}
    initialization = {}
    for mode in args.modes:
        if MODES[mode][0]:
            start = perf_counter()
            indexes[mode] = TaskReuse(dataset.name, history, aliases=aliases, threshold=args.task_threshold)
            initialization[mode] = perf_counter() - start
    with path.open("a", encoding="utf-8", buffering=1) as report:
        for number, task in enumerate(tasks):
            rotation = (number + args.seed) % len(args.modes)
            for mode in args.modes[rotation:] + args.modes[:rotation]:
                if aborted.is_set():
                    return
                index = indexes.get(mode)
                startup = initialization.pop(mode, 0)
                old = previous.get((task["tid"], mode))
                if old is not None:
                    if index is not None and old["verified_answer"] is not None:
                        index.add_verified(task, old["verified_answer"], passed=old["passed"])
                    continue
                result = trial(dataset, task, mode, index, args, startup)
                report.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"{dataset.name} {number + 1}/{len(tasks)} {mode} "
                    f"{'PASS' if result['passed'] else 'FAIL'} "
                    f"requests={result['model_requests']} folds={result['folds']} "
                    f"seconds={result['wall_s']:.2f}", flush=True,
                )


def run_benchmark(args):
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.modes)) != len(args.modes):
        raise ValueError("datasets and modes must be unique")
    if args.limit < 0 or args.workers < 1 or args.rpm < 0 or args.max_turns < 1:
        raise ValueError("workers and turns must be positive, and limit and request rate nonnegative")
    if not (0 <= args.task_threshold <= 1 and 0 <= args.content_threshold <= 1):
        raise ValueError("similarity thresholds must be between 0 and 1")
    sys.path.insert(0, str(args.spartacus))
    from spartacus import provider

    provider.api_key("gemini")
    datasets = [Dataset(name, args.assignment / "assignment2" / "datasets" / name) for name in args.datasets]
    jobs = [(dataset, dataset.read_tasks(args.limit, args.seed)) for dataset in datasets]
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {
        "model": MODEL, "prompt": PROMPT, "seed": args.seed, "modes": args.modes,
        "task_threshold": args.task_threshold, "content_threshold": args.content_threshold,
        "max_turns": args.max_turns, "workers": args.workers, "rpm": args.rpm,
        "tasks": {dataset.name: [task["tid"] for task in tasks] for dataset, tasks in jobs},
    }
    meta = args.out / "settings.json"
    if meta.exists():
        if read_json(meta) != settings:
            raise ValueError("existing results use different experiment settings")
    else:
        pending = meta.with_suffix(".tmp")
        pending.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        pending.replace(meta)
    aborted = Event()
    pace = Pace(args.rpm, aborted)
    instrumented = measured_provider(provider.complete, pace)
    open_request = request.urlopen

    def bounded_request(query, *args, timeout=45, **kwargs):
        return open_request(query, *args, timeout=min(timeout, 45), **kwargs)

    started = perf_counter()
    with patch.object(provider, "complete", instrumented), patch.object(request, "urlopen", bounded_request):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            running = {pool.submit(benchmark_dataset, dataset, tasks, args, aborted): dataset.name
                       for dataset, tasks in jobs}
            try:
                for future in as_completed(running):
                    future.result()
            except (KeyboardInterrupt, Exception):
                aborted.set()
                for future in running:
                    future.cancel()
                raise
    print(f"Benchmark wall clock: {perf_counter() - started:.2f}s", flush=True)


def summarize(args):
    settings = read_json(args.out / "settings.json")
    rows = [row for dataset in settings["tasks"] for row in json_lines(args.out / f"{dataset}.jsonl")]
    completed = {(row["dataset"], row["tid"], row["mode"]): row for row in rows}
    pairs = [(dataset, tid) for dataset, tids in settings["tasks"].items() for tid in tids
             if all((dataset, tid, mode) in completed for mode in settings["modes"])]
    output = {}
    for group in ["all", *settings["tasks"]]:
        output[group] = {}
        for mode in settings["modes"]:
            subset = [completed[dataset, tid, mode] for dataset, tid in pairs
                      if group == "all" or group == dataset]
            counters = ("passed", "tool_calls", "model_requests", "input_tokens", "output_tokens",
                        "wall_s", "rate_wait_s", "task_lookup_s", "task_index_s", "content_s",
                        "folds", "bytes_saved")
            output[group][mode] = {
                "n": len(subset),
                **{name: round(sum(row[name] for row in subset), 4) for name in counters},
                "task_hits": sum(row["reuse_from"] is not None for row in subset),
                "incorrect_task_hits": sum(row["reuse_from"] is not None and not row["passed"] for row in subset),
            }
            if "baseline" in settings["modes"]:
                output[group][mode]["baseline_pass_to_fail"] = sum(
                    completed[row["dataset"], row["tid"], "baseline"]["passed"] and not row["passed"]
                    for row in subset
                )
    print(json.dumps(output, indent=2))


def audit(args):
    """Score a history-only threshold sweep with the unmodified verifiers."""
    thresholds = (0.40, TASK_THRESHOLD, 0.70, 0.85)
    output = {}
    with TemporaryDirectory(prefix="assignment2-audit-") as tmp:
        answer_path = Path(tmp) / "answer.json"
        for dataset_name in args.datasets:
            dataset = Dataset(dataset_name, args.assignment / "assignment2" / "datasets" / dataset_name)
            tasks = dataset.read_tasks(0, SEED)
            index = TaskReuse(dataset_name, dataset.load_history(), aliases=dataset.aliases(), threshold=min(thresholds))
            sweep = {threshold: {"hits": 0, "passed": 0} for threshold in thresholds}
            rejected = []
            for task in tasks:
                match = index.find(task)
                if match:
                    answer, _, score = match
                    answer_path.write_text(json.dumps(answer), encoding="utf-8")
                    passed = dataset.verify(task["tid"], answer_path)
                    for threshold in thresholds:
                        if score >= threshold:
                            sweep[threshold]["hits"] += 1
                            sweep[threshold]["passed"] += passed
                key = identity(dataset_name, task, index.aliases)
                signature = index.hasher.signature(shingles(task_text(dataset_name, task)))
                neighbors = index.index.query(signature)
                if neighbors:
                    neighbor, score = max(neighbors, key=lambda item: (item[1], -item[0]))
                    candidate = index.entries[neighbor]
                    if key is not None and candidate.identity != key and score >= TASK_THRESHOLD:
                        rejected.append((score, task["tid"], candidate))
            hard_negative = None
            for score, tid, candidate in sorted(rejected, key=lambda item: (-item[0], item[1])):
                answer_path.write_text(json.dumps(candidate.answer), encoding="utf-8")
                if not dataset.verify(tid, answer_path):
                    hard_negative = {"tid": tid, "history_id": candidate.source, "score": score}
                    break
            output[dataset_name] = {"n": len(tasks), "thresholds": sweep, "hard_negative": hard_negative}
    print(json.dumps(output, indent=2))


def replay(args):
    """Measure output bytes for fixed read sequences, without agent sampling."""
    sys.path.insert(0, str(args.spartacus))
    from spartacus.tools import core_tools

    fixtures = {"configs": "config/*.yaml", "revisions": "policy/*.md", "logs": "logs/*.log"}
    output = {}
    for name, pattern in fixtures.items():
        dataset = Dataset(name, args.assignment / "assignment2" / "datasets" / name)
        task = dataset.read_tasks(8, SEED)[0]
        with TemporaryDirectory(prefix="assignment2-fixed-reads-") as tmp:
            directory = Path(tmp)
            dataset.stage(task, directory)
            reader = next(tool.run for tool in core_tools(directory) if tool.name == "read_file")
            messages = []
            folder = ContentReuse()
            for file in sorted(directory.glob(pattern)):
                call = {"name": "read_file", "args": {"path": file.relative_to(directory).as_posix()}}
                message = {"name": "read_file", "text": reader(**call["args"])}
                messages.append(message)
                folder.observe(call, message, messages)
            output[name] = {
                "tid": task["tid"], "reads": len(messages), "folds": folder.hits,
                "raw_bytes": sum(len(m["text"].encode()) for m in messages),
                "folded_bytes": sum(len(m["text"].encode()) for m in folder.render(messages)),
            }
    print(json.dumps(output, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    run = actions.add_parser("run", help="run or resume paired Gemini trials")
    run.add_argument("--assignment", type=Path, default=ROOT / "upstream-assignment")
    run.add_argument("--spartacus", type=Path, default=ROOT / "upstream-spartacus")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--datasets", choices=(*DATASETS, *sorted(CORPUS - set(DATASETS))), nargs="+", default=list(DATASETS))
    run.add_argument("--modes", choices=tuple(MODES), nargs="+", default=list(MODES))
    run.add_argument("--limit", type=int, default=0, help="seeded tasks per dataset (0 for all)")
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--rpm", type=float, default=60, help="shared request rate (0 for unpaced)")
    run.add_argument("--max-turns", type=int, default=14)
    run.add_argument("--task-threshold", type=float, default=TASK_THRESHOLD)
    run.add_argument("--content-threshold", type=float, default=CONTENT_THRESHOLD)
    summary = actions.add_parser("summary", help="summarize tasks completed in every mode")
    summary.add_argument("--out", type=Path, required=True)
    sweep = actions.add_parser("audit", help="score history-only matches without model calls")
    sweep.add_argument("--assignment", type=Path, default=ROOT / "upstream-assignment")
    sweep.add_argument("--datasets", choices=(*DATASETS, *sorted(CORPUS - set(DATASETS))), nargs="+",
                       default=list(DATASETS))
    reads = actions.add_parser("replay", help="measure output bytes on fixed read sequences")
    reads.add_argument("--assignment", type=Path, default=ROOT / "upstream-assignment")
    reads.add_argument("--spartacus", type=Path, default=ROOT / "upstream-spartacus")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "run":
            run_benchmark(args)
        elif args.command == "summary":
            summarize(args)
        elif args.command == "audit":
            audit(args)
        else:
            replay(args)
    except (ValueError, RuntimeError, OSError) as error:
        # Provider exceptions can contain requests or diagnostics. Print only the status.
        status = re.search(r"HTTP (\d{3})", str(error))
        if status:
            print(f"Gemini request failed with HTTP {status[1]}. Completed trials were saved.", file=sys.stderr)
        else:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
