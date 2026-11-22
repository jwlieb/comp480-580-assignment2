# COMP 480/580 Assignment 2

This implementation runs the Spartacus agent with optional task-level and content-level MinHash reuse. [REPORT.pdf](REPORT.pdf) describes the method, results, and accuracy trade-offs. [results.json](results.json) contains the experiment settings and per-task measurements.

## Reproduce the experiment

Use Python 3.10 or newer and configure a Gemini key in `~/.config/spartacus/env` or `GEMINI_API_KEY`.

The live results in the report were measured at commit `cc2021dac172577a67a7e2c5fb1f79946d3e8223`. To reproduce that revision in a separate checkout, run:

```sh
git worktree add --detach ../assignment2-benchmark cc2021dac172577a67a7e2c5fb1f79946d3e8223
cd ../assignment2-benchmark
```

Run the commands below from the benchmark checkout for the archived configuration or from this checkout for the reviewed implementation.

```sh
git clone https://github.com/sahiljoshi515/Assignment2_COMP_580.git upstream-assignment
git -C upstream-assignment checkout --detach a7c5076fec5150934565adb228411b19070070d3
git clone https://github.com/sahiljoshi515/Spartacus.git upstream-spartacus
git -C upstream-spartacus checkout --detach 711db625b3513a34c6521010ea64a35635825b5a

python3 policy.py run --out runs/main --limit 8 --workers 8 --rpm 96
python3 policy.py summary --out runs/main
python3 policy.py audit
python3 policy.py replay
```

The run selects eight tasks per dataset with seed 480580 and evaluates each task in all four configurations. Repeat the same command to continue an interrupted run. Omit `--limit 8` and use a fresh output directory to evaluate every incoming task. Use `--datasets` to select datasets or `--modes` to select configurations. The audit scores history matches with the supplied verifiers. The replay measures compression on fixed tool reads.

The reviewed implementation sends full output when the same tool arguments identify different results. Citation reuse distinguishes a trailing `et al.` from an author name and compares years exactly. The offline audit and fixed-read replay reproduced the same results after these changes.

## Implementation

- `minhash.py` implements three-word shingles, 128 seeded hashes, and LSH candidate search.
- `policy.py` checks task equivalence, indexes verified answers, and runs the paired experiment.
- `content.py` builds reconstructible line edits against an earlier full tool result.
