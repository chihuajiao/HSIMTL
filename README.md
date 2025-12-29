# MSUANet: Multi-Stage Information Sharing Multitask Hyperspectral Classification Network with Unmixing Assistance

This repository provides the official implementation of **MSUANet**, a novel multitask learning framework designed for hyperspectral image (HSI) analysis. MSUANet simultaneously performs **HSI classification** and **spectral unmixing**, integrating a multi-stage cross-task information sharing mechanism. By leveraging the physical interpretability from the unmixing task, the model enhances discriminative feature learning, thereby significantly improving classification performance.

<figure align="center">
  <img width="100%" alt="MSUANet Architecture"
       src="https://github.com/user-attachments/assets/71cf1a34-75c4-46d1-9d20-570e76201e52" />
  <figcaption>
    <b>Figure 1.</b> Overall architecture of MSUANet for HSI classification and spectral unmixing.
  </figcaption>
</figure>


## 🛠️ Environment Setup
### Requirements
```bash
# Create conda environment
conda create -n MSUANet python=3.8
conda activate MSUANet

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install scipy scikit-learn matplotlib spectral tensorboard tqdm
```
### Core Dependencies
- Python >= 3.7
- PyTorch >= 1.12
- scipy
- scikit-learn
- matplotlib
- spectral
- tensorboard
- numpy


## 📊 Dataset Preparation
| Dataset | Classes | Bands | Resolution |
|---------|---------|-------|------------|
| DC (Washington DC Mall) | 7 | 191 | 1280×307 |
| IP (Indian Pines) | 16 | 200 | 145×145 |
| HU (Houston) | 15 | 144 | 349×1905 |
| BE (Berlin) | 8 | 244 | 1723×476 |
### Directory Structure
```
dataset/
├── DC/
│   ├── dc_um.mat        # HSI data + endmembers
│   └── DC_gt.mat        # Ground truth labels
├── IP/
│   ├── IP_edata.mat
│   └── Indian_pines_gt.mat
├── HU/
│   ├── Houston_em.mat
│   └── Houston_gt.mat
└── ...
```

## 🚀 Quick Start
### Basic Usage
```bash
# Train on Houston dataset with 20 samples per class
python main.py --dataset HU --train_num 20 --epoch 300 --seed 2333
```
### Using Shell Script (Batch Experiments)

```bash
# Run experiments on multiple datasets
bash main.sh
```

## 📝 Citation

If you find this code useful for your research, please consider citing:

```bibtex
@article{msuanet2024,
  title={MSUANet: Multi-Stage Information Sharing Multitask Hyperspectral Classification Network with Unmixing Assistance},
  author={Zhang, Mingyang and Xie, Yuhang and Liu, Hao and Wu, Shuang and Yang, Bufang and Jiang, Fenlong and Zhou, Yu and Gong, Maoguo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE},
  note={doi: {10.1109/TGRS.2025.3649827}}
}
```

## 📧 Contact
For questions and issues, please open a GitHub issue or contact [yhxie2022@163.com].
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Acknowledgment—The authors would like to thank the authors of the open-source implementations that contributed to this work. Parts of our code were adapted from the following repositories: 
- [CyCU-Net](https://github.com/hanzhu97702/IEEE_TGRS_CyCU-Net)
- [DSNet](https://github.com/hanzhu97702/DSNet)
- [FreeNet](https://github.com/Z-Zheng/FreeNet)
We sincerely appreciate their valuable efforts in making their code publicly available.
