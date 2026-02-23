## Countering Multi-modal Representation Collapse through Rank-targeted Fusion (WACV 2026)

### Links
- Project page: https://skvictoria.github.io/papers/wacv2025 :contentReference[oaicite:5]{index=5}  
- Paper (arXiv): https://arxiv.org/abs/2511.06450 :contentReference[oaicite:6]{index=6}

This repository provides the official code and experiments for **“Countering Multi-modal Representation Collapse through Rank-targeted Fusion.”** :contentReference[oaicite:0]{index=0}

Multi-modal fusion often suffers from two coupled failure modes:
- **Feature collapse**: representation diversity shrinks as variation concentrates in only a few directions (e.g., in the eigenspectrum).
- **Modality collapse**: one dominant modality overwhelms the other, reducing balanced multi-modal reasoning. :contentReference[oaicite:1]{index=1}

We propose **effective rank** as a unified measure to quantify and counter both collapses, and introduce **Rank-enhancing Token Fuser**, a theoretically grounded fusion method that selectively blends less-informative features from one modality with complementary features from another to increase the effective rank of the fused representation. :contentReference[oaicite:2]{index=2}  
To further address modality collapse, we analyze modality pairings and show that **depth** helps preserve representational balance when fused with RGB. :contentReference[oaicite:3]{index=3}

We validate the approach on **human action anticipation** and present **R3D**, a depth-informed fusion framework, demonstrating improvements across **NTU RGB+D, UTKinect, and DARai** (up to **+3.74%** over prior SOTA). :contentReference[oaicite:4]{index=4}


### Citation
```bibtex
@inproceedings{kim2026ranktargetedfusion,
  title     = {Countering Multi-modal Representation Collapse through Rank-targeted Fusion},
  author    = {Kim, Seulgi and Kokilepersaud, Kiran and Prabhushankar, Mohit and AlRegib, Ghassan},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
