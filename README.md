# PFLinMedicine

This is the official implementation of **"Personalised Federated Learning with Asymmetric Client Collaboration"**


## Quick Start Cifar100
- generate Cifar100 dataset
    ```
    cd .\dataset\
    python generate_Cifar100.py noniid - dir  
    ```
- run PFL-AC/Lay-PFL-AC on Cifar100
    ```
    cd .\system\
    python main.py --data Cifar100 --algo PFL-AC --global_lr 0.02 --ar_lr 0.05 --num_classes 100 --model CNN
    python main.py --data Cifar100 --algo Lay-PFL-AC --global_lr 0.05 --ar_lr 0.02 --num_classes 100 --model CNN
    ```

## Quick Start MIDOG++
- download MIDOG++ dataset
    ```
    cd .\dataset\
    python download_MIDOG.py
    ```
- generate MIDOGpp dataset
    ```
    python generate_MIDOG.py
    ```

- run PFL-A/Lay-PFL-AC on MIDOGpp 
    ```
    cd .\system\
    python main.py --data MIDOGpp --algo PFL-AC --global_lr 0.02 --ar_lr 0.05 --num_classes 7 --model CNN --num_clients 7
    python main.py --data MIDOGpp --algo Lay-PFL-AC --global_lr 0.05 --ar_lr 0.02 --num_classes 7 --model CNN --num_clients 7    
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
