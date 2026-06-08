# Model-free-Computational-Optics
>This repo is an opensource library for **model-free computational optics**, which optimizes the optical systems  without requiring explicit numerical models.

<p align="center"><img src="assets/mfo_conceptual_plot.png" style="width:70%; border:0;"></p>

### 1. Computational Optics Optimization Genres

| Genres | Core | Strength | Weakness |
| --- | --- | --- | --- |
|**Whitebox(MBO)** | Phyiscs model | Fast; physics grounding | sim2real gap |
|**Graybox(MBO)** |  Phyiscs model + Real system feedback | High fidelity | Model Bias/ Computation burden |
|**Blackbox(MFO)** | Real system feedback | Medium fielity; easy to be effective | More hardware queries |

### 2. We have examined  the following tasks:

| Optical computing | Computer-generated holography (CGH)|
| --- | --- |
| <img src="assets/optical_computing_aigc.png" alt="AI-generated optical computing task illustration" width="420"> | <img src="assets/cgh_aigc.png" alt="AI-generated computer-generated holography task illustration" width="420"> |

The task illustrations are AI-generated assets. The prompts and OpenAI API script are in [utils/generate_readme_aigc_images.py](utils/generate_readme_aigc_images.py).

#### a. MFO for Optical Computing

<span style="color:magenta">Best paper award of ICCP 2024</span>

*ICCP&TPAMI 2024* | [Project page](https://shuxin626.github.io/mfo_optical_computing/index.html) | [Arxiv](https://arxiv.org/abs/2307.11957) | [Guangyuan Zhao](https://zhaoguangyuan123.github.io), [Xin Shu](), [Renjie Zhou](https://www.renjiezhou.com/)


#### b. MFO for CGH

*TENCON 2022* | [Paper](); [Guangyuan Zhao](https://zhaoguangyuan123.github.io), [Renjie Zhou](https://www.renjiezhou.com/)

#### c. To be continued ... 



### 3. Run details

See [docs/details.md](docs/details.md) for optical computing (1- and 2-layer simulators) CGH (simulator) run instructions.

### 4. TODO

- [x] Further refactor repo.
- [x] Add detailed simulator-based two-layer optical computing code.
- [x] Add CGH run code.
- [ ] More MFO algorithms :construction:.
- [ ] More comptutational optics tasks :construction:.

### 5. Related publications

Not only us that pushing the direction of model-free computational optics: 

>a. [Li et al., Model-free optical processors using in situ reinforcement learning with proximal policy optimization](https://www.nature.com/articles/s41377-025-02148-7)
Direct followup work that used PPO as core algo for model-free computational optics.
b. [More will come ...]()




### 6. Citation

If you find our work useful, please cite our paper:

```bibtex
@article{zhao2024high,
  title={High-performance real-world optical computing trained by in situ gradient-based model-free optimization},
  author={Zhao, Guangyuan and Shu, Xin and Zhou, Renjie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
```bibtex
@inproceedings{zhao2022model,
  title={Model-free computer-generated holography},
  author={Zhao, Guangyuan and Zhou, Renjie},
  booktitle={TENCON 2022-2022 IEEE Region 10 Conference (TENCON)},
  pages={1--3},
  year={2022},
  organization={IEEE}
}
```
