# CNN/ — TTConv2d experiments (historical, Phase A of the project)

Synthetic edge-CNN demo of Tensor-Train convolution (`ttcompress.ttconv.TTConv2d`:
im2col → TTLinear → fold). Status: **Implemented, tested** (`tests/test_core.py::test_ttconv_matches_conv`).
Results here are on *synthetic* weights with a hand-built decaying spectrum and do not
transfer to real CNNs (see `docs/history/REAL_MODEL_FINDINGS.md`, Findings 1–4).

- `demo_cnn_pipeline.py` — synthetic pipeline (writes `results_cnn.json`)
- `real_cnn_example.py` — TT on a real torchvision/timm CNN
- `make_cnn_report_plots.py` — plots from `results_cnn.json`

Note: `results_cnn.json` has uncommitted changes from an unrecorded re-run (RESEARCH_AUDIT.md M7);
its latency numbers are single-run measurements and are not used anywhere in the current analysis.
