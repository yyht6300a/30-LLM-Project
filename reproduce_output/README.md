# reproduce_output

Raw output of `scripts/reproduce.sh cpu` on TC2 (2026-11-13), kept as reproduction evidence. No code reads this directory.

| Subdirectory | Contents | Compared against |
|---|---|---|
| `p1/` | P1 statistics and bootstrap logs, `verify_core_artifacts.log` | paper numbers in `reference/Popular-but-Wrong` |
| `calibrator_eval/` | regenerated P2 evaluation report | `calibrator/eval/output/` |
| `artifacts/` | the 18 MLP calibrators retrained from scratch (`.pt` / `.npz` / `.json`) | `calibrator/training/artifacts/` |

`artifacts/` has the same file names as `calibrator/training/artifacts/` but different contents. The latter are the
weights P2 committed and the MCP service loads; the files here were retrained from the same code to check that those
weights can be reproduced. The two differ only at floating-point level: 14 of the 18 combinations have identical test
Alignment, and each of the other 4 differs by one test question. See the "TC2 复现结果" section of
[scripts/README.md](../scripts/README.md) for the numbers.
