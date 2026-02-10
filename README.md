# PFLinMedicine

This is the official implementation of **"FedCASE: Quantifying Asymmetric Collaboration for Personalized Federated Learning"**


## Quick Start
- generate Cifar100 dataset 
    ```
    cd .\dataset\
    python generate_Cifar100.py noniid - dir  
    ```
-run FedCASE/Lay-FedCASE
    ```
    cd .\system\
    python main.py --data Cifar100 --algo Lay-FedCASE --global_lr 0.05 --ar_lr 0.02
    ```


🎯**This repository is an extension of the following paper：**

```
@article{zhang2023pfllib,
  title={PFLlib: Personalized Federated Learning Algorithm Library},
  author={Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian},
  journal={arXiv preprint arXiv:2312.04992},
  year={2023}
}
```
