## Countering Multi-modal Representation Collapse through Rank-targeted Fusion (WACV 2026)

### Links
- Paper (arXiv): https://arxiv.org/abs/2511.06450

This repository provides the official code and experiments for **“Countering Multi-modal Representation Collapse through Rank-targeted Fusion.”**

Multi-modal fusion often suffers from two coupled failure modes:
- **Feature collapse**: representation diversity shrinks as variation concentrates in only a few directions.
- **Modality collapse**: one dominant modality overwhelms the other, reducing balanced multi-modal reasoning.

### TL;DR
We propose effective rank as a unified measure to quantify and counter both collapses, and introduce Rank-enhancing Token Fuser, a theoretically grounded fusion method that selectively blends less-informative features from one modality with complementary features from another to increase the effective rank of the fused representation.
To further address modality collapse, we analyze modality pairings and show that depth helps preserve representational balance when fused with RGB.
We validate the approach on human action anticipation / action segmentation, demonstrating improvements across diverse datasets.


### Citation
```bibtex
@inproceedings{kim2026ranktargetedfusion,
  title     = {Countering Multi-modal Representation Collapse through Rank-targeted Fusion},
  author    = {Kim, Seulgi and Kokilepersaud, Kiran and Prabhushankar, Mohit and AlRegib, Ghassan},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
