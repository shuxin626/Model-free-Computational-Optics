# Model-free-Computational-Optics
This repo is an opensource library for **model-free computational optics**. 

Model-free optimization optimizes the optical systems $f_{sys}(x, w)$ without requiring explicit numerical models $\hat{f}_{sys}(x, w)$.

<p align="center"><img src="imgs/mfo_conceptual_plot.png" style="width:70%; border:0;"></p>

### 1. This repo contains the following tasks:

#### - Model-free Optimization for Optical Computing
High-performance real-world optical computing trained by in situ gradient-based model-free optimization 

🏆 <span style="color:magenta">Best paper award of ICCP 2024</span>

*ICCP&TPAMI 2024* | [Project page](https://shuxin626.github.io/mfo_optical_computing/index.html) | [Arxiv](https://arxiv.org/abs/2307.11957) | 
[Guangyuan Zhao](https://zhaoguangyuan123.github.io), [Xin Shu](), [Renjie Zhou](https://www.renjiezhou.com/)


#### - Model-free Optimization for Computer Generated Holography
Model-free computer generated holography
Zhao, Guangyuan, and Renjie Zhou. TENCON 2022-2022 IEEE Region 10 Conference (TENCON). IEEE, 2022.


#### - To be continued...

### 2. How to start (example for optical computing):

1. **Clone this repo**

2. **Choose an optimization method for training** in `param/param_onn.py`:
    ```python
    settings['optimizer'] = 'mfo' # our method
    settings['optimizer'] = 'sbt' # simulator-based method
    settings['optimizer'] = 'hbt' # hybrid training method
    ```

3. **Start training**

   ```bash
   python main_onn.py
   ```

4. **Set parameters for testing** in `param/param_onn.py`:

    ```python
    settings['train_or_test'] = 'test'
    test_param['ckpt_dir] = YOUR_CHECKPOINT_DIR
    ```

5. **Start testing**

    ```bash
    python main_onn.py
    ```

### 3. Citation

If you find our work useful, please cite our paper:
<!-- ```bibtex
@article{zhao2023modelfreeopticalcomputing,
  title={High-performance real-world optical computing trained by in situ model-free optimization},
  author={Zhao, Guangyuan and Shu, Xin and Zhou, Renjie},
  journal={arXiv preprint arXiv:2307.11957},
  year={2023}
}
``` -->
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

