# PFLinMedicine

This is the official implementation of **"FedCASE: Quantifying Asymmetric Collaboration for Personalized Federated Learning"**.

## Datasets and scenarios
***Medical datasets***: [**MedMNISTT**](https://medmnist.com/), [**MIDOG++**](https://github.com/DeepMicroscopy/MIDOGpp).


***General datasets***: 
For the ***class imbalance*** scenario, we introduce **14** famous datasets: **MNIST**, **EMNIST**, **Fashion-MNIST**, **Cifar10**, **Cifar100**, **AG News**, **Sogou News**, **Tiny-ImageNet**, **Country211**, **Flowers102**, **GTSRB**, **Shakespeare**, and **Stanford Cars**, they can be easy split into **IID** and **non-IID** version. Since some codes for generating datasets such as splitting are the same for all datasets, we move these codes into `./dataset/utils/dataset_utils.py`. In the **non-IID** scenario, 2 situations exist. The first one is the **pathological non-IID** scenario, the second one is the **practical non-IID** scenario. In the **pathological non-IID** scenario, for example, the data on each client only contains the specific number of labels (maybe only 2 labels), though the data on all clients contains 10 labels such as the MNIST dataset. In the **practical non-IID** scenario, Dirichlet distribution is utilized (please refer to this [paper](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html) for details). We can input `balance` for the iid scenario, where the data are uniformly distributed. 

For the ***domain shift*** scenario, we use **4** datasets that are widely used in Domain Adaptation: **Amazon Review** (fetch raw data from [this site](https://drive.google.com/file/d/1QbXFENNyqor1IlCpRRFtOluI2_hMEd1W/view?usp=sharing)), **Digit5** (fetch raw data from [this site](https://drive.google.com/file/d/1sO2PisChNPVT0CnOvIgGJkxdEosCwMUb/view)), **PACS**, and **DomainNet**.

*If you need another data set, just write another code to download it and then use the utils.*



### Quick Start
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
