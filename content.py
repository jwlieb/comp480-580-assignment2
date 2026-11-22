"""Fold repeated read results into lossless, trajectory-local line edits."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from time import perf_counter

from minhash import LSH, MinHash, shingles

READ_TOOLS = {"read_file", "grep", "list_files"}
THRESHOLD = 0.65


@dataclass(frozen=True)
class Observation:
    message: dict
    label: str
    lines: tuple
    numbered: bool


def _split(tool, output):
    lines = tuple(output.split("\n"))
    if tool == "read_file" and all(
        line.startswith(f"{index}\t") for index, line in enumerate(lines, 1)
    ):
        return tuple(line.split("\t", 1)[1] for line in lines), True
    return lines, False


def _join(lines, numbered):
    if numbered:
        return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, 1))
    return "\n".join(lines)


def changes(base, current):
    """Return ordered (one-based start, delete count, new lines) edits."""
    matcher = SequenceMatcher(a=base, b=current, autojunk=False)
    return [
        (start + 1, end - start, current[new_start:new_end])
        for op, start, end, new_start, new_end in matcher.get_opcodes()
        if op != "equal"
    ]


def reconstruct(base, edits):
    cursor = 0
    result = []
    for start, removed, inserted in edits:
        offset = start - 1
        result.extend(base[cursor:offset])
        result.extend(inserted)
        cursor = offset + removed
    result.extend(base[cursor:])
    return result


class ContentReuse:
    def __init__(self, *, threshold=THRESHOLD, hasher=None):
        self.threshold = threshold
        self.hasher = hasher if hasher is not None else MinHash()
        self.index = LSH()
        self.observations = []
        self.folded = {}
        self.source_contents = {}
        self.seconds = 0.0
        self.hits = 0
        self.bytes_saved = 0

    def observe(self, call, message, messages):
        """Prepare an optional delta for the most recent tool message."""
        start = perf_counter()
        try:
            self._observe(call, message, messages)
        finally:
            self.seconds += perf_counter() - start

    def render(self, messages):
        """Keep the raw transcript available when a reference leaves the window."""
        if not self.folded:
            return messages
        start = perf_counter()
        try:
            rendered = []
            visible = set()
            for message in messages:
                fold = self.folded.get(id(message))
                if fold is not None and id(fold[1].message) in visible:
                    rendered.append({**message, "text": fold[2]})
                else:
                    rendered.append(message)
                visible.add(id(message))
            return rendered
        finally:
            self.seconds += perf_counter() - start

    def _observe(self, call, message, messages):
        tool, output = call["name"], message["text"]
        if tool not in READ_TOOLS:
            return
        label = f'{tool} {json.dumps(call.get("args", {}), ensure_ascii=False, sort_keys=True)}'
        # A repeated command can observe a different version. Its arguments
        # alone can no longer identify an earlier result unambiguously.
        if self.source_contents.setdefault(label, output) != output:
            self.source_contents[label] = None
        if len(output) < 400 or len(output) > 80_000:
            return
        if output.startswith(("ERROR:", "BLOCKED:")) or "... truncated at " in output:
            return
        lines, numbered = _split(tool, output)
        if len(lines) > 600:
            return
        signature = self.hasher.signature(shingles("\n".join(lines), mask_numbers=True))
        live = {id(item) for item in messages}
        candidates = sorted(self.index.query(signature), key=lambda item: (-item[1], item[0]))
        original_size = len(output.encode())
        shortest = None
        best_base = None
        for key, score in candidates[:8]:
            base = self.observations[key]
            if score < self.threshold or id(base.message) not in live:
                continue
            if base.message["name"] != tool or base.numbered != numbered:
                continue
            if self.source_contents[base.label] is None:
                continue
            edits = changes(base.lines, lines)
            if _join(reconstruct(base.lines, edits), numbered) != output:
                continue
            line_numbers = " Renumber the resulting lines from 1." if numbered else ""
            payload = json.dumps(edits, ensure_ascii=False, separators=(",", ":"))
            delta = (
                f"[reuse] Current: {label}. Base: earlier full result for "
                f"{base.label}. Keep all unchanged lines. Edits are "
                "[base line (1-based), delete count, exact new lines]."
                f"{line_numbers}\n{payload}"
            )
            if shortest is None or len(delta.encode()) < len(shortest.encode()):
                shortest = delta
                best_base = base
        if shortest is not None and len(shortest.encode()) <= original_size * 0.75:
            self.folded[id(message)] = (message, best_base, shortest)
            self.hits += 1
            self.bytes_saved += original_size - len(shortest.encode())
            return

        self.index.add(len(self.observations), signature)
        self.observations.append(Observation(message, label, lines, numbered))
