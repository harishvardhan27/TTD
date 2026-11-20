# Literature Review

**Verification method (2026-11-20).** Each entry's title, authors, year and
venue were checked against an official proceedings/publisher page or arXiv
listing (URLs below). Technique and finding summaries are taken from the
abstract / official page; where a summary relies on secondary sources it is
marked *(secondary)*. Experimental details not visible in those sources are
not claimed. Full texts were not all read; entries are therefore summaries at
abstract level unless stated otherwise.

---

## 1. Foundations

| # | Reference | Venue / ID | Relevance |
|---|---|---|---|
| F1 | C. Eckart, G. Young. *The approximation of one matrix by another of lower rank.* | Psychometrika 1(3):211–218, 1936. doi:[10.1007/BF02288367](https://doi.org/10.1007/BF02288367) | optimal truncated SVD (§1 of derivations) |
| F2 | I. V. Oseledets. *Tensor-Train Decomposition.* | SIAM J. Sci. Comput. 33(5):2295–2317, 2011. doi:[10.1137/090752286](https://epubs.siam.org/doi/10.1137/090752286) | TT-SVD, quasi-optimality bound |
| F3 | A. Novikov, D. Podoprikhin, A. Osokin, D. Vetrov. *Tensorizing Neural Networks.* | NeurIPS 28 (2015), pp. 442–450. [proceedings](https://papers.nips.cc/paper/5787-tensorizing-neural-networks), arXiv:1509.06569 | TT-format fully-connected layers: the project's original method |
| F4 | E. Denton, W. Zaremba, J. Bruna, Y. LeCun, R. Fergus. *Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation.* | NeurIPS 27 (2014). [proceedings](https://proceedings.neurips.cc/paper/2014/hash/2afe4567e1bf64d32a5527244d104cea-Abstract.html) | early SVD/tensor compression for inference speed; ~2× conv speed-up within 1% accuracy |
| F5 | G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* | arXiv:[1503.02531](https://arxiv.org/abs/1503.02531), 2015 (NIPS 2014 DL Workshop) | temperature-scaled KD loss used in Phase 4 |
| F6 | B. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* | CVPR 2018, pp. 2704–2713. doi:10.1109/CVPR.2018.00286, [CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html) | INT8 affine quantization scheme underlying torch/ORT INT8 |

## 2. Data-aware / activation-aware low-rank factorisation

| # | Reference | Venue | Problem → technique | Relevant findings (abstract level) | Difference from this project |
|---|---|---|---|---|---|
| D1 | P. Chen, H.-F. Yu, I. Dhillon, C.-J. Hsieh. *DRONE: Data-aware Low-rank Compression for Large NLP Models.* | NeurIPS 2021. [poster](https://neurips.cc/virtual/2021/poster/28707), [PDF](https://papers.nips.cc/paper_files/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf) | BERT weights are not low-rank, but layer representations lie in a low-dimensional space → provably optimal **closed-form data-aware** low-rank decomposition for FC and attention layers | 1.92× faster on MRPC with 1.5% accuracy loss without fine-tuning; >12× with distillation | Same objective family as our §2 (minimise output error on data). NLP (BERT); GPU. **Our data-aware SVD is not new.** |
| D2 | H. Yu, J. Wu. *Compressing Transformers: Features Are Low-Rank, but Weights Are Not!* | AAAI 2023, 37(9):11007–11015. doi:[10.1609/aaai.v37i9.26304](https://ojs.aaai.org/index.php/AAAI/article/view/26304) | transformer weights near full rank, output features low-rank → compress layer outputs (AAFM) then global fine-tuning (GFM) | covers ViTs and language models | Directly anticipates our observation that *plain* SVD fails on ViT weights while activation-aware compression works better. |
| D3 | Y.-C. Hsu, T. Hua, S. Chang, Q. Lou, Y. Shen, H. Jin. *Language Model Compression with Weighted Low-rank Factorization* (FWSVD). | ICLR 2022. [poster](https://iclr.cc/virtual/2022/poster/6157), arXiv:2207.00112 | Fisher-information-weighted SVD aligned with task loss | lower task-accuracy loss though not lower reconstruction error | Uses gradient (Fisher) importance; we use activation second moments only (no labels/gradients). A Fisher-weighted variant is future work. |
| D4 | Z. Yuan, Y. Shang, Y. Song, Q. Wu, Y. Yan, G. Sun. *ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs.* | arXiv:[2312.05821](https://arxiv.org/abs/2312.05821), 2023 (v5 2025; venue not verified) | scale weights by activation statistics (diagonal) before SVD; per-layer sensitivity-based rank | 10–20% (later 10–30%) compression of LLMs training-free | Diagonal scaling ≈ a special case of our full-covariance whitening. |
| D5 | X. Wang, Y. Zheng, Z. Wan, M. Zhang. *SVD-LLM: Truncation-aware Singular Value Decomposition for LLM Compression.* | ICLR 2025. [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3104e1ab39875cf54fe1eb4473e7c5a1-Abstract-Conference.html), arXiv:2403.07378 | **truncation-aware data whitening** (Cholesky of activation Gram matrix) so singular values map directly to compression loss; sequential low-rank parameter update | whitening is the main contributor *(secondary)* | **Our `factorize(W, cov)` is the SVD-LLM whitening** (Cholesky of `XXᵀ`, SVD of `W L`, `L⁻¹` folded into `V`). We add explicit damping and its interpretation (derivations §2.1), and held-out error reporting. |

## 3. Low-rank compression of Vision Transformers

| # | Reference | Venue | Technique | Findings | Difference |
|---|---|---|---|---|---|
| V1 | J. Xiao, M. Yin, Y. Gong, X. Zang, J. Ren, B. Yuan. *COMCAT: Towards Efficient Compression and Customization of Attention-Based Vision Models.* | ICML 2023, PMLR 202:38125–38136. [PMLR](https://proceedings.mlr.press/v202/xiao23e.html) | head-level low-rankness in MHA; automatic rank selection; MHA + FFN | at equal compression rate, SVD gave higher accuracy without fine-tuning than Tucker or Tensor-Train, with better GPU throughput *(secondary, from search summary of the paper)* | **Already establishes SVD > TT for ViTs** — our B6 comparison replicates, not discovers, this on CPU. |
| V2 | S. Azizi, M. Nazemi, M. Pedram. *Memory-Efficient Vision Transformers: An Activation-Aware Mixed-Rank Compression Strategy.* | ECCV 2024 Workshops, doi:10.1007/978-3-031-91979-4_6; arXiv:[2402.06004](https://arxiv.org/abs/2402.06004) | activation-aware, each weight as sum of two low-rank matrices, mixed ranks | DeiT-B −60% params with <1% ImageNet drop (with their pipeline) | **Activation-aware + mixed-rank for DeiT already exists.** |
| V3 | M. Thoma et al. *FLAR-SVD: Fast and Latency-Aware Singular Value Decomposition for Model Compression.* | CVPR 2025 Workshops, pp. 1889–1898, doi:10.1109/CVPRW67362.2025.00178. [CVF](https://openaccess.thecvf.com/content/CVPR2025W/MAI/html/Thoma_FLAR-SVD_Fast_and_Latency-Aware_Singular_Value_Decomposition_for_Model_Compression_CVPRW_2025_paper.html) | stabilised Cholesky SVD, fast rank search, **latency tuning with a learned latency predictor** (random-forest regressor) | DeiT 81.0% ImageNet after 1 epoch fine-tune, −30% latency | **Hardware/latency-aware rank selection for ViTs already exists.** Our §3.4 can at most replicate it on a different CPU. |

## 4. Low-rank + quantization

| # | Reference | Venue | Technique | Difference |
|---|---|---|---|---|
| Q1 | Z. Yao, X. Wu, C. Li, S. Youn, Y. He. *ZeroQuant-V2* (LoRC). | arXiv:[2303.08302](https://arxiv.org/abs/2303.08302), 2023 | low-rank factors approximate the *quantization error* `W − Q(W)`; factors storable in 8-bit | Low-rank as an error *corrector* for quantization; we use low-rank as the *primary* compression and INT8 on top. |
| Q2 | O. Gordon et al. *MLoRQ: Bridging Low-Rank and Quantization for Transformer Compression.* | BMVC 2025. [proceedings](https://bmvc2025.bmva.org/proceedings/668/), arXiv:2507.09616 | joint per-layer choice of rank and bit-width under a memory constraint; ViTs | **Joint low-rank + quantization allocation for ViTs already exists** and is more general (mixed precision) than our fixed INT8. |

## 5. Evaluation data and models

| # | Reference | Venue | Use here |
|---|---|---|---|
| E1 | B. Recht, R. Roelofs, L. Schmidt, V. Shankar. *Do ImageNet Classifiers Generalize to ImageNet?* | ICML 2019, PMLR 97:5389–5400. [ICML](https://icml.cc/virtual/2019/poster/4270) | ImageNetV2 test set; models lose 11–14 pp vs ImageNet-val |
| E2 | H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, H. Jégou. *Training data-efficient image transformers & distillation through attention.* | ICML 2021, PMLR 139:10347–10357. [PMLR](https://proceedings.mlr.press/v139/touvron21a.html) | DeiT-S model |
| E3 | T. Ridnik, E. Ben-Baruch, A. Noy, L. Zelnik-Manor. *ImageNet-21K Pretraining for the Masses.* | NeurIPS 2021 Datasets & Benchmarks. [proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract-round1.html) | source of the ViT-B "miil" checkpoint |

---

## 6. Positioning: what is established vs. what this project adds

| Claim area | Established in literature | Implemented here | Possible contribution (must be supported by our data) | Not demonstrated / needed to claim |
|---|---|---|---|---|
| Whitened (data-aware) SVD beats weight SVD for transformers | D1, D2, D5 (NLP/LLM), V2 (ViT) | yes (`lowrank.factorize`) | none as a method; an **independent CPU replication on ImageNetV2 with paired statistics** | — |
| SVD beats TT for ViTs | V1 | yes (B6) | replication at matched parameter budget with output-error accounting | — |
| Global rank allocation | ALDS ([Liebenwein et al., NeurIPS 2021](https://arxiv.org/abs/2107.11442), min-max error), D4, V2, V3 | greedy + exact DP (multiple-choice knapsack) | a measured greedy-vs-exact gap; explicit bound (5.2) | any accuracy benefit of DP over greedy |
| Damping as interpolation between data-aware and plain SVD | regularisation is common (V3 "stabilised Cholesky"); interpretation (2.6) not found in the verified abstracts | yes | an ablation of λ on ViTs with the interpretation | literature search not exhaustive: **do not claim novelty** |
| Latency-aware ranks | V3 | not yet (Phase 5 decides) | CPU (AVX2, no VNNI) measurements of when factorisation pays off | — |
| Low-rank + INT8 on CPU | Q1, Q2 | yes (torch + ORT; operator coverage verified) | **measured** interaction on an AVX2 laptop CPU incl. operation order (FQ vs QF) and ORT graph coverage | — |
| Rigorous evaluation protocol | — | contamination audit, ImageNetV2 dev/test, McNemar, bootstrap latency | a cautionary finding: Imagenette / sample-image sets used in prior project versions overlap ImageNet-train | — |

**Conclusion of the review.** None of the *techniques* is new. The defensible
contributions are empirical and methodological: (i) a controlled, statistically
tested comparison of plain vs whitened SVD, TT, INT8 and their combinations for
two ViTs on a contamination-checked held-out set; (ii) CPU deployment
measurements on a common AVX2 laptop processor, including ONNX Runtime operator
coverage; (iii) documented negative results. The manuscript's title and claims
are scoped accordingly.

*Not covered (acknowledged gap):* structured pruning and token pruning for
ViTs; post-training quantization methods beyond dynamic INT8 (e.g. PTQ4ViT,
RepQ-ViT); LLM-scale follow-ups (SVD-LLM V2). These were not verified and are
not cited.
