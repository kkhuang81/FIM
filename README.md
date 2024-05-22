## Scalable Continuous-time Diffusion Framework for Network Inference and Influence Estimation

**FIM** is framework to approximate the continuous-time diffusion from cascade data. It is established by utilizing the continuous-time dynamical system. In this paper, **FIM** is proposed for network inference from cascades and influence estimation on the inferred networks. 

This repository contains the source codes of **FIM**. For further details, please refer to our paper in **WWW 2024** (https://arxiv.org/abs/2403.02867). Should you encounter any issues, please reach out to Keke Huang, thanks!



## Requirements

- CUDA 10.1.243
- python 3.6.10
- pytorch 1.4.0
- GCC 5.4.0
- [cnpy](https://github.com/rogersce/cnpy)
- [swig-4.0.1](https://github.com/swig/swig)



## Compilation

```
make
```



## Command example

```
python train.py --seed 25190 --dataset HR --steps 2500 --patience 1000 --lr 0.001 --l1_lambda 0.01 --batch 50 --tau 0.5 --norm 0.5  --train_per 0.8 --val_per 0.1
```





## Citation



Please cite our paper if it is relevant to your work, thanks!

```
@inproceedings{HuangGCX24,
  author       = {Keke Huang and
                  Ruize Gao and
                  Bogdan Cautis and
                  Xiaokui Xiao},
  title        = {Scalable Continuous-time Diffusion Framework for Network Inference
                  and Influence Estimation},
  booktitle    = {{WWW}},
  pages        = {2660--2671},
  year         = {2024},
}
```

