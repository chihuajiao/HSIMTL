import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random

import torch
import torch.utils.data
from torch.utils.data import dataset
import scipy.io as sio
import torchvision.transforms as transforms
import torch.nn.functional as F

class HUSPLIT(torch.utils.data.Dataset):
    def __init__(self, img, gt, transform=None, tile_size=(128, 128)):
        self.img = img.float()  # Assume img is already a tensor
        self.gt = gt.float()    # Assume gt is already a tensor
        self.transform = transform
        self.tile_size = tile_size
        self.tiles = self._generate_tiles()

    def _generate_tiles(self):
        # Generate tiles for the entire dataset
        tiles = []
        num_tiles_h = self.img.shape[1] // self.tile_size[0]  # Horizontal tiles
        num_tiles_w = self.img.shape[2] // self.tile_size[1]  # Vertical tiles

        for h in range(num_tiles_h):
            for w in range(num_tiles_w):
                img_tile = self.img[:, h * self.tile_size[0]:(h + 1) * self.tile_size[0],
                                    w * self.tile_size[1]:(w + 1) * self.tile_size[1]]
                gt_tile = self.gt[:, h * self.tile_size[0]:(h + 1) * self.tile_size[0],
                                  w * self.tile_size[1]:(w + 1) * self.tile_size[1]]
                tiles.append((img_tile, gt_tile))
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_tile, gt_tile = self.tiles[idx]

        # Check if transformation is required and if img_tile is not a tensor
        if self.transform and not isinstance(img_tile, torch.Tensor):
            img_tile = self.transform(img_tile)

        return img_tile, gt_tile

class MyTrainData(torch.utils.data.Dataset):
  def __init__(self, img, gt, transform=None):
    self.img = img.float()
    self.gt = gt.float()
    self.transform=transform

  def __getitem__(self, idx):
    return self.img,self.gt

  def __len__(self):
    return 1

class SpectralTransform(torch.nn.Module):
    """对高光谱数据进行随机翻转、旋转、噪声注入和随机裁剪"""
    def __init__(self, flip_prob=0.5, rotate_prob=0.5,  mask_prob=0.1, mask_value=0,mask_size=1, mask_num=50):
        super(SpectralTransform, self).__init__()
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.mask_prob = mask_prob
        self.mask_value = mask_value
        self.mask_size = mask_size
        self.mask_num  = mask_num
        

    def forward(self, x):
        # 随机翻转
        if np.random.rand() < self.flip_prob:
            x = x.flip(-1)  # 水平翻转
        if np.random.rand() < self.flip_prob:
            x = x.flip(-2)  # 垂直翻转
        # 随机旋转
        if np.random.rand() < self.rotate_prob:
            k = np.random.randint(0, 4)
            x = x.rot90(k, dims=(-2, -1))  # 旋转
        # 添加掩码
        if np.random.rand() < self.mask_prob:
            for _ in range(int(np.random.rand() * self.mask_num)):  # 随机生成掩码数量
                mask_x = random.randint(0, x.shape[-2] - self.mask_size)
                mask_y = random.randint(0, x.shape[-1] - self.mask_size)
                x[:, mask_x:mask_x + self.mask_size, mask_y:mask_y + self.mask_size] = self.mask_value
        return x


def predict_data_loader(x,y,patch_size,pca_components,BATCH_SIZE,NUM_WORKER):

    print('Hyperspectral data shape: ', x.shape)
    print('Label shape: ', y.shape)
    non_zero_indices = np.where(y != 0)
    print(non_zero_indices)

    print('\n... ... PCA tranformation ... ...')
    x_pca = apply_pca(x, num_components=pca_components)
    print('Data shape after PCA: ', x_pca.shape)

    print('\n... ... create data cubes ... ...')
    x_pca, y = create_image_cubes(x_pca, y, window_size=patch_size)
    print('Data cube X shape: ', x_pca.shape)
    print('Data cube y shape: ', y.shape)


    x_predict = x_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', x_predict.shape)

    x_predict = x_predict.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', x_predict.shape)
    
    predict_set = TestDS(x_predict, y)
    
    test_loader = torch.utils.data.DataLoader(dataset=predict_set,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=NUM_WORKER)

    return test_loader,y ,non_zero_indices


#   预测的数据结果
def create_data_loader(x,y,test_ratio,patch_size,pca_components,BATCH_SIZE,NUM_WORKER):

    print('Hyperspectral data shape: ', x.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    x_pca = apply_pca(x, num_components=pca_components)
    print('Data shape after PCA: ', x_pca.shape)

    print('\n... ... create data cubes ... ...')
    x_pca, y = create_image_cubes(x_pca, y, window_size=patch_size)
    print('Data cube X shape: ', x_pca.shape)
    print('Data cube y shape: ', y.shape)

    # 按比例取出数据
    print('\n... ... create train & test data ... ...')
    x_train, x_test, y_train, y_test = split_train_test_set_ratio(x_pca, y, test_ratio)
    print('Xtrain shape: ', x_train.shape)
    print('Xtest  shape: ', x_test.shape)
    
    
    # 打印测试集标签类别及其数量
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print("Label classes and their counts in the test set:")
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} instances")
        
    

    # 改变 x_train, y_train 的形状，以符合 keras 的要求
    x_train = x_train.reshape(-1, patch_size, patch_size, pca_components, 1)
    x_test = x_test.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', x_train.shape)
    print('before transpose: Xtest  shape: ', x_test.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    x_train = x_train.transpose(0, 4, 3, 1, 2)
    x_test = x_test.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', x_train.shape)
    print('after transpose: Xtest  shape: ', x_test.shape)

    # 创建 train_loader 和 test_loader
    train_set = TrainDS(x_train, y_train, transform=False)#SpectralTransform(0.5, 0.5)
    test_set = TestDS(x_test, y_test)
    
    
    
    
    # 数据集的加载
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=NUM_WORKER)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=NUM_WORKER)
    
    # 计算每类标签的数量
    label_counts = {}
    for _, labels in train_loader:
        unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label in label_counts:
                label_counts[label] += count
            else:
                label_counts[label] = count

    # 打印每类标签的数量
    print("Label classes and their counts after augmentation:")
    for label, count in label_counts.items():
        print(f"Class {label}: {count} instances")

    return train_loader, test_loader, y_test


# 对高光谱数据 x 应用 PCA 变换
def apply_pca(x, num_components):
    new_x = np.reshape(x, (-1, x.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_x = pca.fit_transform(new_x)
    new_x = np.reshape(new_x, (x.shape[0], x.shape[1], num_components))
    return new_x

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padding_with_zeros(x, margin=2):
    new_x = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2 * margin, x.shape[2]))
    x_offset = margin
    y_offset = margin
    new_x[x_offset:x.shape[0] + x_offset, y_offset:x.shape[1] + y_offset, :] = x
    return new_x

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def create_image_cubes(x, y, window_size=5, remove_zero_labels=True):
    # 给 x 做 padding
    margin = int((window_size - 1) / 2) 
    zero_padded_x = padding_with_zeros(x, margin=margin)
    # split patches
    patches_data = np.zeros((x.shape[0] * x.shape[1], window_size, window_size, x.shape[2]))
    patches_labels = np.zeros((x.shape[0] * x.shape[1]))
    patch_index = 0
    for r in range(margin, zero_padded_x.shape[0] - margin):
        for c in range(margin, zero_padded_x.shape[1] - margin):
            patch = zero_padded_x[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patches_labels[patch_index] = y[r-margin, c-margin]
            patch_index += 1
    if remove_zero_labels:
        patches_data = patches_data[patches_labels > 0, :, :, :]
        patches_labels = patches_labels[patches_labels > 0]
        patches_labels -= 1
    return patches_data, patches_labels

def split_train_test_set_ratio(x, y, test_ratio, random_state=345):
    x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                        test_size=test_ratio,
                                                        random_state=random_state,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test


def split_train_test_set_perclass(x, y, num_samples_per_class, random_state=345):
    """
    Split the dataset into training and test sets with manual selection of training samples per class.

    Parameters:
    x (array): Input features.
    y (array): Target labels.
    num_samples_per_class (dict): Number of samples to select for training from each class.
    random_state (int): Random state for reproducibility.

    Returns:
    tuple: Arrays containing the split datasets (x_train, x_test, y_train, y_test).
    """
    np.random.seed(random_state)
    unique_classes = np.unique(y)
    x_train, x_test, y_train, y_test = [], [], [], []

    for cls in unique_classes:
        class_idx = np.where(y == cls)[0]
        np.random.shuffle(class_idx)
        train_idx = class_idx[:num_samples_per_class[cls]]
        test_idx = class_idx[num_samples_per_class[cls]:]

        x_train.extend(x[train_idx])
        y_train.extend(y[train_idx])
        x_test.extend(x[test_idx])
        y_test.extend(y[test_idx])

    # Converting lists to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Optionally shuffle the training set to mix up classes
    train_indices = np.arange(x_train.shape[0])
    np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    return x_train, x_test, y_train, y_test

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):

    def __init__(self, x_test, y_test):

        self.len = x_test.shape[0]
        self.x_data = torch.FloatTensor(x_test)
        self.y_data = torch.LongTensor(y_test)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):

    def __init__(self, x_train, y_train, transform=None):

        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)
        self.y_data = torch.LongTensor(y_train)
        self.transform = transform

    def __getitem__(self, index):

        x = self.x_data[index]
        y = self.y_data[index]

        # 如果定义了transform，则应用它
        if self.transform:
            
            x = self.transform(x)

        return x, y

    def __len__(self):

        # 返回文件数据的数目
        return self.len