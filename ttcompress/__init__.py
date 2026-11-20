"""ttcompress: Tensor-Train and low-rank compression of neural-network layers.

Modules
-------
tt        TT-SVD, HOSVD diagnostics, TTLinear
ttconv    TTConv2d (im2col -> TTLinear)
surgery   swap nn.Linear / nn.Conv2d for TT layers in any nn.Module
lowrank   LowRankLinear, plain and data-aware SVD, budgeted rank allocation
calib     input-covariance collection for data-aware SVD
data      real ImageNet-labelled images for CPU evaluation (cached)
evaluate  accuracy with confidence intervals and paired tests
bench     interleaved latency benchmarking
quant     INT8 (torch dynamic, ONNX Runtime) baselines
distill   knowledge-distillation recovery fine-tuning
"""

from .lowrank import LowRankLinear, compress_lowrank, factorize
from .tt import TTLinear, tt_reconstruct, tt_svd
from .ttconv import TTConv2d

__version__ = "0.2.0"
