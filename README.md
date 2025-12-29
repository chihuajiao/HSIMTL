# MSUANet: Multi-Stage Information Sharing Multitask Hyperspectral Classification Network with Unmixing Assistance

This repository provides the official implementation of **MSUANet**, a novel multitask learning framework designed for hyperspectral image (HSI) analysis. MSUANet simultaneously performs **HSI classification** and **spectral unmixing**, integrating a multi-stage cross-task information sharing mechanism. By leveraging the physical interpretability from the unmixing task, the model enhances discriminative feature learning, thereby significantly improving classification performance.

<figure align="center">
  <img width="100%" alt="MSUANet Architecture"
       src="https://github.com/user-attachments/assets/71cf1a34-75c4-46d1-9d20-570e76201e52" />
  <figcaption>
    <b>Figure 1.</b> Overall architecture of MSUANet for HSI classification and spectral unmixing.
  </figcaption>
</figure>

## 📋 Table of Contents

- [Abstract](#Abstract)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Citation](#citation)

## 🎯 Abstract
Multitask learning (MTL) can effectively exploit the complementary information across different tasks in hyperspectral images (HSIs) to support classification. However, existing MTL methods for HSI classification usually employ shared structures in the early stages or simple information fusion in the later stages, failing to adequately characterize task specific feature differences and adaptively select task-relevant information. Therefore, we propose a multi-stage information sharing multitask hyperspectral classification network with unmixing assistance (MSUANet). The proposed method adopts a three stage strategy, including low-level information sharing, task-specific information mining, and adaptive latent information selection, to fully exploit the information gain introduced by the unmixing task and provide more fine-grained spectral details for improved classification performance. In addition, we propose an information diverter with hybrid attention (IDHA) module, which enhances task-specific information for classification and unmixing, respectively. Furthermore, we design the adaptive latent abundance knowledge transfer (ALAKT) module, which adaptively selects latent complementary information for both tasks. Experimental results on Indian Pines, Houston 2013, Berlin, and Washington DC Mall datasets demonstrate that MSUANet outperforms state-of-the-art methods.

## 🛠️ Environment Setup
### Requirements
```bash
# Create conda environment
conda create -n MSUANet python=3.8
conda activate MSUANet

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
# Install dependencies
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
### Supported Datasets
| Dataset | Classes | Bands | Resolution |
|---------|---------|-------|------------|
| DC (Disease Crop) | 7 | 191 | 1280×307 |
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
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 📧 Contact

For questions and issues, please open a GitHub issue or contact [yhxie2022@163.com].

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
