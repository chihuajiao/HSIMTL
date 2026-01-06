<h1 align="center">MSUANet</h1>

<p align="center">
  <b>Multi-Stage Information Sharing Multitask Hyperspectral Classification Network with Unmixing Assistance</b>
</p>

<p align="center">
  <img width="80%" alt="MSUANet Architecture"
       src="https://github.com/user-attachments/assets/71cf1a34-75c4-46d1-9d20-570e76201e52" />
</p>

<p align="left">
  This repository provides the official implementation of <b>MSUANet</b>, a multitask learning framework for hyperspectral image (HSI) analysis.
</p>


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

### Dataset 
We conduct experiments on four widely used hyperspectral datasets: Washington DC Mall (DC), Indian Pines (IP), Houston (HU), and Berlin (BE).
The dataset has been uploaded and you can download it here  BaiDu Cloud drive:(https://pan.baidu.com/s/1yNS0DektpZ-3K7XvQ9wfhQ code: 1234).

Regarding the creation of the data de-mixing part, you can refer to the introduction in the paper or the [Q1 question](#q1) in the Q&A section.


| Dataset | Classes | Bands | Resolution |
|---------|---------|-------|------------|
| DC (Washington DC Mall) | 7 | 191 | 1280×307 |
| IP (Indian Pines) | 16 | 200 | 145×145 |
| HU (Houston) | 15 | 144 | 349×1905 |
| BE (Berlin) | 8 | 244 | 1723×476 |

### Dataset Directory 
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

## 🚀 Training

Our codebase has integrated several widely used comparison methods, including **UNet, FreeNet, SSFCN, FContNet, UperNet, SegFormer, and TransUNet**.  


If needed, you may conveniently supplement or reproduce the comparison experiments by running the following commands, which helps reduce the additional effort required for implementation and configuration.
### Train MSUANet
```bash
# Train on Houston dataset with 20 samples per class
python main.py --dataset HU --train_num 20 --epoch 300 --seed 2333
```
### Train Compare Methods
```bash
python compare_main.py --dataset HU --net TransUNet --train_num 20 --epoch 300 --seed 2333
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
## 🙏 Acknowledgments

Acknowledgment—The authors would like to thank the authors of the open-source implementations that contributed to this work. Parts of our code were adapted from the following repositories: 
- [CyCU-Net](https://github.com/hanzhu97702/IEEE_TGRS_CyCU-Net)
- [DSNet](https://github.com/hanzhu97702/DSNet)
- [FreeNet](https://github.com/Z-Zheng/FreeNet)

We sincerely appreciate their valuable efforts in making their code publicly available.

## 📧 Contact
For questions and issues, please open a GitHub issue or contact [yhxie2022@163.com].

## ❤️ Final Words
Thank you for your interest in our work.  
We hope this repository will be helpful to your research, and we wish you every success in your scientific journey.

## Q&A

### Q1: How to make rough and unmixed labels
<a id="q1"></a>
A1: The rough unmixing labels allow you to select the area of interest, and then use methods such as FCLSU to obtain the results based on the endmember curves and the actual hyperspectral image. You can refer to the process shown in the following picture.
[fig5_HU_plot.pdf](https://github.com/user-attachments/files/24453275/fig5_HU_plot.pdf)

