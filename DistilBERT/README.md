# DistilBERT/ — TTLinear on transformer-FFN shapes (historical)

Synthetic DistilBERT-FFN-shaped stack (768↔3072) compressed with `ttcompress.tt.TTLinear`,
plus an optional HuggingFace example (`hf_real_model_example.py`, needs `transformers` + internet).
Status: **Implemented, tested on synthetic weights only.** The founding assumption (FFN weights are
low-rank) was refuted on real ViT weights; see `docs/history/REAL_MODEL_FINDINGS.md`, Finding 5.
`results.json` carries an unrecorded re-run (RESEARCH_AUDIT.md M7) and is not used in the current analysis.
