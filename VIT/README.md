# VIT/ — TT vs plain SVD on real ViT-B weights (historical)

`real_vit_test.py` compares 3-mode TT and plain SVD reconstruction error on real ViT-B layers at the
*same rank number* (not matched parameters, contrary to the old root README). The matched-budget
comparison is now part of the main grid (`configs/deit_small.yaml`, variants `tt_*`).
