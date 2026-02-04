# PFLinMedicine

This is the official implementation of **"FedCASE: Quantifying Asymmetric Collaboration for Personalized Federated Learning"**


## Quick Start
- MIDOG++ 
    ```
    cd ./dataset
    python generate_MIDOG.py
    ```

The output of `python generate_MIDOG.py`
```
load canine_cutaneous_mast_cell_tumor
load canine_lung_cancer
load canine_lymphosarcoma
load canine_soft_tissue_sarcoma
load human_breast_cancer
load human_melanoma
load human_neuroendocrine_tumor
Total number of samples: 26273
The number of train samples: [2760, 1354, 6162, 2745, 3326, 1556, 1800]
The number of test samples: [920, 452, 2054, 916, 1109, 519, 600]

Saving to disk.

Finish generating dataset.
</details>



🎯**This repository is an extension of the following paper：**

```
@article{zhang2023pfllib,
  title={PFLlib: Personalized Federated Learning Algorithm Library},
  author={Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian},
  journal={arXiv preprint arXiv:2312.04992},
  year={2023}
}
```
