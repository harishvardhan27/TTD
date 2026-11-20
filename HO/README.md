# HO/ — the superseded 'finalized scope' (historical)

`low_rank_compress.py` and `validate_finalized_project.py` produced the old claim
"1.2x compression, 2–3% drop, 12% faster" (`docs/history/HO_FINALIZED_PROJECT.md`).
The audit found that the rank search silently fell back to rank 256 (C2), the evaluation used 100
training-provenance images (C1), and the latency claim came from 5 sequential iterations (H7).
Both files are kept unchanged (apart from a header note and the pinned checkpoint tag) so that the
finding can be reproduced. Use `ttcompress.lowrank` + `experiments/run.py` instead.
