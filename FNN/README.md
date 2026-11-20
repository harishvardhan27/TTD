# FNN/ — first real-model validation (historical)

MobileNetV2 / ResNet34 with real ImageNet weights; TT conv compression and a 12-step distillation.
The findings write-up is now `docs/history/REAL_MODEL_FINDINGS.md`.

Caveat (RESEARCH_AUDIT.md C1): the 200 evaluation images come from the EliSchwartz sample set,
which is of ImageNet-*train* provenance; accuracies here are not held-out test accuracies.
The current, held-out protocol uses ImageNetV2 (`docs/EXPERIMENTAL_PROTOCOL.md`).
