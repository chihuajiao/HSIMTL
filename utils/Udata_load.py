import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.io import loadmat
import torch.nn.functional as F
import matplotlib.pyplot as plt

def fixed_num_sample(gt_mask: np.ndarray, num_train_samples, num_classes, seed=2333):
    """
    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: int
        seed: int

    Returns:
        train_indicator, test_indicator: 2-D arrays of shape [height, width]
    """
    rs = np.random.RandomState(seed)
    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten, dtype=np.bool_)
    test_indicator = np.zeros_like(gt_mask_flatten, dtype=np.bool_)
    
    for cls in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == cls)[0]
        rs.shuffle(inds)
        
        if len(inds) < num_train_samples:
            raise ValueError(f"Class {cls} has fewer samples ({len(inds)}) than num_train_samples ({num_train_samples}).")
        
        train_inds = inds[:num_train_samples]
        test_inds = inds[num_train_samples:]
        
        train_indicator[train_inds] = True
        test_indicator[test_inds] = True
    
    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)
    
    return train_indicator, test_indicator

def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, minibatch_size, seed):
    """
    Args:
        gt_mask: 2-D array of shape [height, width]
        train_indicator: 2-D array of shape [height, width]
        minibatch_size: int
        seed: int

    Returns:
        List of 2-D arrays indicating selected training indices for each minibatch
    """
    rs = np.random.RandomState(seed)
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = {}
    
    for cls in cls_list:
        if cls == 0:
            continue  # 假设类标签从1开始，0为背景
        train_inds_per_class = (gt_mask == cls) & (train_indicator == 1)
        inds = np.where(train_inds_per_class.ravel())[0]
        rs.shuffle(inds)
        inds_dict_per_class[cls] = inds
    
    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            left = cnt * minibatch_size
            if left >= len(inds):
                continue
            right = min((cnt + 1) * minibatch_size, len(inds))
            fetch_inds = inds[left:right]
            train_inds[fetch_inds] = 1
        cnt += 1
        if train_inds.sum() == 0:
            break
        train_inds_list.append(train_inds.reshape(train_indicator.shape))
    
    return train_inds_list


def divisible_pad(tensors, divisor, mode='constant', value=0):
    """
    Pads each tensor in the list so that height and width are divisible by `divisor`.
    All tensors must have the same H and W before padding.
    
    Args:
        tensors: List of numpy arrays with shape [C, H, W] or [1, H, W]
        divisor: int
        mode: Padding mode ('constant', 'reflect', etc.)
        value: Padding value if mode is 'constant'
        
    Returns:
        List of padded tensors
    """
    # 确保所有张量的高度和宽度相同
    shapes = [tensor.shape for tensor in tensors]
    if not all(shape[1:] == shapes[0][1:] for shape in shapes):
        raise ValueError("All tensors must have the same height and width before padding.")
    
    _, H, W = tensors[0].shape
    pad_h = (divisor - H % divisor) if H % divisor != 0 else 0
    pad_w = (divisor - W % divisor) if W % divisor != 0 else 0
    
    padded_tensors = []
    for tensor in tensors:
        
        padded = F.pad(torch.tensor(tensor), (0, pad_w, 0, pad_h), mode=mode, value=value)
        padded_tensors.append(padded.numpy())
    
    return padded_tensors


class NewDCDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=50,
                 sub_minibatch=10,
                 divisor=16,
                 seed=2333):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        self._load_data() 
        # self._preprocess() 
        self._split_data() 
        self._pad_data() 
        if self.training: 
            self.train_inds_list = minibatch_sample(
                self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
                seed=self.rs.randint(0, 2**32 - 1)
            )
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
        
        # self.image = im_mat['DC']  # 假设键名为 'paviaU'
        im_mat = torch.from_numpy(im_mat['Y']).float()
        # print('1111111111111',im_mat.shape)
        im_mat = im_mat.reshape(191, 307,1280).permute(0, 2, 1)  # (matlab row frist,but python is not)
        # print('222222222222222222',im_mat.shape)
        
        im_mat = im_mat.permute(1, 2, 0)
        # print(im_mat.shape)
        
        self.image = im_mat
        # self.save_as_pseudo_color_image()
        
        
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = torch.from_numpy(gt_mat['DCgt']).long()  # mat's keys
        
        print('hsi data load',self.image.shape)
        print('hsi mask load',self.mask.shape)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    # def _preprocess(self):
    #     # 计算每个通道的均值和标准差
    #     C = self.image.shape[2]
    #     self.im_cmean = self.image.reshape(-1, C).mean(axis=0)
    #     self.im_cstd = self.image.reshape(-1, C).std(axis=0)
        
    #     # 归一化
    #     self.image = (self.image - self.im_cmean) / self.im_cstd
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        print('pading maks',self.image.shape)
        # image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        image_np = self.image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        # 更新类属性，不执行填充
        self.padded_image = image_np  # [C, H, W]
        self.padded_mask = mask_np  # [1, H, W]
        if self.training:
            self.padded_train_indicator = train_indicator_np  # [1, H, W]
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
        else:
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask':  self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
            }
        else:
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask': self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
            }

    def save_as_pseudo_color_image(self):
        # 选择三个通道用于 RGB
        if self.image.shape[0] < 3:
            raise ValueError("Not enough channels in the image")
        # 归一化每个通道
        r = self.image[:, :, 10]
        g = self.image[:, :, 30]
        b = self.image[:, :, 50]
        r_normalized = (r - r.min()) / (r.max() - r.min())
        g_normalized = (g - g.min()) / (g.max() - g.min())
        b_normalized = (b - b.min()) / (b.max() - b.min())

        # 合并通道制作彩色图像
        rgb_image = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)
        
        # 使用 matplotlib 保存伪彩色图像
        plt.imsave('pseudo_color_image.png', rgb_image)
        print("Pseudo-color image saved as 'pseudo_color_image.png'")

class NewIPDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=5,
                 sub_minibatch=1,
                 divisor=16,
                 seed=2333):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        self._load_data() # 加载数据
        # self._preprocess() # 数据预处理
        self._split_data() # 数据切分
        self._pad_data() # 填充
        if self.training: # 生成训练小批量
            self.train_inds_list = minibatch_sample(
                self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
                seed=self.rs.randint(0, 2**32 - 1)
            )
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
        im_mat = torch.from_numpy(im_mat['Y'].astype(np.float32)).float()
        im_mat = im_mat.reshape(200, 145,145).permute(0, 2, 1)  # (matlab row frist,but python is not)
        im_mat = im_mat.permute(1, 2, 0)
        
        self.image = im_mat
        # self.save_as_pseudo_color_image()
        
        
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = torch.from_numpy(gt_mat['indian_pines_gt']).long()  # mat's keys
        
        print('hsi data load',self.image.shape)
        print('hsi mask load',self.mask.shape)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    # def _preprocess(self):
    #     # 计算每个通道的均值和标准差
    #     C = self.image.shape[2]
    #     self.im_cmean = self.image.reshape(-1, C).mean(axis=0)
    #     self.im_cstd = self.image.reshape(-1, C).std(axis=0)
        
    #     # 归一化
    #     self.image = (self.image - self.im_cmean) / self.im_cstd
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        print('pading maks',self.image.shape)
        # image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        image_np = self.image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        # 更新类属性，不执行填充
        self.padded_image = image_np  # [C, H, W]
        self.padded_mask = mask_np  # [1, H, W]
        if self.training:
            self.padded_train_indicator = train_indicator_np  # [1, H, W]
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
        else:
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask':  self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
            }
        else:
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask': self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
            }

    def save_as_pseudo_color_image(self):
        # 选择三个通道用于 RGB
        if self.image.shape[0] < 3:
            raise ValueError("Not enough channels in the image")
        # 归一化每个通道
        r = self.image[:, :, 10]
        g = self.image[:, :, 30]
        b = self.image[:, :, 50]
        r_normalized = (r - r.min()) / (r.max() - r.min())
        g_normalized = (g - g.min()) / (g.max() - g.min())
        b_normalized = (b - b.min()) / (b.max() - b.min())

        # 合并通道制作彩色图像
        rgb_image = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)
        
        # 使用 matplotlib 保存伪彩色图像
        plt.imsave('pseudo_color_image.png', rgb_image)
        print("Pseudo-color image saved as 'pseudo_color_image.png'")

class NewHUDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=5,
                 sub_minibatch=1,
                 divisor=16,
                 seed=2333):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        self._load_data() # 加载数据
        # self._preprocess() # 数据预处理
        self._split_data() # 数据切分
        self._pad_data() # 填充
        if self.training: # 生成训练小批量
            self.train_inds_list = minibatch_sample(
                self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
                seed=self.rs.randint(0, 2**32 - 1)
            )
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
        im_mat = torch.from_numpy(im_mat['Y'].astype(np.float32)).float()
        im_mat = im_mat.reshape(144, 1905,349).permute(0, 2, 1)  # (matlab row frist,but python is not)
        im_mat = im_mat.permute(1, 2, 0)

        self.image = im_mat
        # self.save_as_pseudo_color_image()
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = torch.from_numpy(gt_mat['Houston_gt']).long()  # mat's keys
        
        print('hsi data load',self.image.shape)
        print('hsi mask load',self.mask.shape)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        print('pading maks',self.image.shape)
        # image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        image_np = self.image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        # 更新类属性，不执行填充
        self.padded_image = image_np  # [C, H, W]
        self.padded_mask = mask_np  # [1, H, W]
        if self.training:
            self.padded_train_indicator = train_indicator_np  # [1, H, W]
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
        else:
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask':  self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
            }
        else:
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask': self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
            }

    def save_as_pseudo_color_image(self):
        if self.image.shape[0] < 3:
            raise ValueError("Not enough channels in the image")
        
        r = self.image[:, :, 10]
        g = self.image[:, :, 30]
        b = self.image[:, :, 50]
        r_normalized = (r - r.min()) / (r.max() - r.min())
        g_normalized = (g - g.min()) / (g.max() - g.min())
        b_normalized = (b - b.min()) / (b.max() - b.min())

        rgb_image = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)

        plt.imsave('pseudo_color_image.png', rgb_image)
        print("Pseudo-color image saved as 'pseudo_color_image.png'")

class NewPaviaDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=200,
                 sub_minibatch=10,
                 divisor=16,
                 seed=2333):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        # 加载数据
        self._load_data()
        
        # 数据预处理
        # self._preprocess()
        
        # 数据切分
        self._split_data()
        
        # 填充
        self._pad_data()
        
        # 生成训练小批量
        if self.training:
            self.train_inds_list = minibatch_sample(
                self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
                seed=self.rs.randint(0, 2**32 - 1)
            )
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
        self.image = im_mat['paviaU']  # 假设键名为 'paviaU'
        
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = gt_mat['paviaU_gt']  # 假设键名为 'paviaU_gt'
        
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    # def _preprocess(self):
    #     # 计算每个通道的均值和标准差
    #     C = self.image.shape[2]
    #     self.im_cmean = self.image.reshape(-1, C).mean(axis=0)
    #     self.im_cstd = self.image.reshape(-1, C).std(axis=0)
        
    #     # 归一化
    #     self.image = (self.image - self.im_cmean) / self.im_cstd
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        if self.training:
            tensors_to_pad = [image_np, mask_np, train_indicator_np, test_indicator_np]
        else:
            tensors_to_pad = [image_np, mask_np, test_indicator_np]
        
        # 调用修改后的 divisible_pad 函数
        padded_tensors = divisible_pad(tensors_to_pad, self.divisor, mode='constant', value=0)
        
        self.padded_image = padded_tensors[0]  # [C, H_padded, W_padded]
        self.padded_mask = padded_tensors[1]  # [H_padded, W_padded]
        if self.training:
            self.padded_train_indicator = padded_tensors[2]  # [H_padded, W_padded]
            self.padded_test_indicator = padded_tensors[3]  # [H_padded, W_padded]
        else:
            self.padded_test_indicator = padded_tensors[2]  # [H_padded, W_padded]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            return {
                'image': torch.tensor(self.padded_image, dtype=torch.float32),  # [C, H, W]
                'mask': torch.tensor(self.padded_mask, dtype=torch.long),  # [H, W]
                'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
            }
        else:
            return {
                'image': torch.tensor(self.padded_image, dtype=torch.float32),  # [C, H, W]
                'mask': torch.tensor(self.padded_mask, dtype=torch.long),  # [H, W]
                'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
            }

class NewPUDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=5,
                 sub_minibatch=1,
                 divisor=16,
                 seed=2333):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        self._load_data() # 加载数据
        # self._preprocess() # 数据预处理
        self._split_data() # 数据切分
        self._pad_data() # 填充
        if self.training: # 生成训练小批量
            self.train_inds_list = minibatch_sample(
                self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
                seed=self.rs.randint(0, 2**32 - 1)
            )
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
        im_mat = torch.from_numpy(im_mat['Y'].astype(np.float32)).float()
        im_mat = im_mat.reshape(103,340,610).permute(0, 2, 1)  # (matlab row frist,but python is not)
        im_mat = im_mat.permute(1, 2, 0)

        self.image = im_mat
        self.save_as_pseudo_color_image()
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = torch.from_numpy(gt_mat['paviaU_gt']).long()  # mat's keys
        
        print('hsi data load',self.image.shape)
        print('hsi mask load',self.mask.shape)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        print('pading maks',self.image.shape)
        # image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        image_np = self.image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        # 更新类属性，不执行填充
        self.padded_image = image_np  # [C, H, W]
        self.padded_mask = mask_np  # [1, H, W]
        if self.training:
            self.padded_train_indicator = train_indicator_np  # [1, H, W]
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
        else:
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask':  self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
            }
        else:
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask': self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
            }

    def save_as_pseudo_color_image(self):
        if self.image.shape[0] < 3:
            raise ValueError("Not enough channels in the image")
        
        r = self.image[:, :, 10]
        g = self.image[:, :, 30]
        b = self.image[:, :, 50]
        r_normalized = (r - r.min()) / (r.max() - r.min())
        g_normalized = (g - g.min()) / (g.max() - g.min())
        b_normalized = (b - b.min()) / (b.max() - b.min())

        rgb_image = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)

        plt.imsave('pseudo_color_image.png', rgb_image)
        print("Pseudo-color image saved as 'pseudo_color_image.png'")

class NewBEDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=5,
                 sub_minibatch=1,
                 divisor=16,
                 seed=2333):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        self._load_data() # 加载数据
        # self._preprocess() # 数据预处理
        self._split_data() # 数据切分
        self._pad_data() # 填充
        if self.training: # 生成训练小批量
            self.train_inds_list = minibatch_sample(
                self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
                seed=self.rs.randint(0, 2**32 - 1)
            )
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
        im_mat = torch.from_numpy(im_mat['Y'].astype(np.float32)).float()
        im_mat = im_mat.reshape(244,476,1723).permute(0, 2, 1)  # (matlab row frist,but python is not)
        im_mat = im_mat.permute(1, 2, 0)

        self.image = im_mat
        self.save_as_pseudo_color_image()
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = torch.from_numpy(gt_mat['gt']).long()  # mat's keys
        
        print('hsi data load',self.image.shape)
        print('hsi mask load',self.mask.shape)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        print('pading maks',self.image.shape)
        # image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        image_np = self.image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        # 更新类属性，不执行填充
        self.padded_image = image_np  # [C, H, W]
        self.padded_mask = mask_np  # [1, H, W]
        if self.training:
            self.padded_train_indicator = train_indicator_np  # [1, H, W]
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
        else:
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask':  self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
            }
        else:
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask': self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
            }

    def save_as_pseudo_color_image(self):
        if self.image.shape[0] < 3:
            raise ValueError("Not enough channels in the image")
        
        r = self.image[:, :, 10]
        g = self.image[:, :, 30]
        b = self.image[:, :, 50]
        r_normalized = (r - r.min()) / (r.max() - r.min())
        g_normalized = (g - g.min()) / (g.max() - g.min())
        b_normalized = (b - b.min()) / (b.max() - b.min())

        rgb_image = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)

        plt.imsave('pseudo_color_image.png', rgb_image)
        print("Pseudo-color image saved as 'pseudo_color_image.png'")

class NewLKDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=5,
                 sub_minibatch=1,
                 divisor=16,
                 seed=2333):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        self._load_data() # 加载数据
        # self._preprocess() # 数据预处理
        self._split_data() # 数据切分
        self._pad_data() # 填充
        if self.training: # 生成训练小批量
            self.train_inds_list = minibatch_sample(
                self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
                seed=self.rs.randint(0, 2**32 - 1)
            )
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
        im_mat = torch.from_numpy(im_mat['Y'].astype(np.float32)).float()
        im_mat = im_mat.reshape(270,400,550).permute(0, 2, 1)  # (matlab row frist,but python is not)
        im_mat = im_mat.permute(1, 2, 0)

        self.image = im_mat
        self.save_as_pseudo_color_image()
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = torch.from_numpy(gt_mat['WHU_Hi_LongKou_gt']).long()  # mat's keys
        
        print('hsi data load',self.image.shape)
        print('hsi mask load',self.mask.shape)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        print('pading maks',self.image.shape)
        # image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        image_np = self.image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        # 更新类属性，不执行填充
        self.padded_image = image_np  # [C, H, W]
        self.padded_mask = mask_np  # [1, H, W]
        if self.training:
            self.padded_train_indicator = train_indicator_np  # [1, H, W]
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
        else:
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask':  self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
            }
        else:
            return {
                'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                'mask': self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
                'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
            }

    def save_as_pseudo_color_image(self):
        if self.image.shape[0] < 3:
            raise ValueError("Not enough channels in the image")
        
        r = self.image[:, :, 102]
        g = self.image[:, :, 130]
        b = self.image[:, :, 104]
        r_normalized = (r - r.min()) / (r.max() - r.min())
        g_normalized = (g - g.min()) / (g.max() - g.min())
        b_normalized = (b - b.min()) / (b.max() - b.min())

        rgb_image = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)

        plt.imsave('pseudo_color_image.png', rgb_image)
        print("Pseudo-color image saved as 'pseudo_color_image.png'")

class NewHUselfDataset(Dataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=5,
                 sub_minibatch=5,
                 divisor=16,
                 seed=2333,
                 labeled = True):
        """
        Args:
            image_mat_path: str, path to the image .mat file
            gt_mat_path: str, path to the ground truth .mat file
            training: bool, whether the dataset is for training or testing
            num_train_samples_per_class: int, number of training samples per class
            sub_minibatch: int, number of samples per minibatch per class
            divisor: int, for padding to make dimensions divisible by this number
            seed: int, random seed for reproducibility
        """
        self.image_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.divisor = divisor
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        self.labeled = labeled
        
        self._load_data()          # 加载数据
        # self._preprocess()       # 数据预处理
        self._split_data()         # 数据切分
        self._pad_data()           # 填充
        if self.training:          # 生成训练小批量
            self.train_inds_list = minibatch_sample(self.padded_mask,
                                                    self.padded_train_indicator,
                                                    self.sub_minibatch,
                                                    seed=self.rs.randint(0, 2**32 - 1))
        else:
            self.test_inds_list = [0]
    
    def _load_data(self):
        im_mat = loadmat(self.image_mat_path)
 
        # self.image = im_mat['DC']  # 假设键名为 'paviaU'
        # print('Original data type:', im_mat['Y'].dtype)
        im_mat = torch.from_numpy(im_mat['Y'].astype(np.float32)).float()
        # print('1111111111111',im_mat.shape)
        im_mat = im_mat.reshape(144, 1905,349).permute(0, 2, 1)  # (matlab row frist,but python is not)
        # print('222222222222222222',im_mat.shape)
        im_mat = im_mat.permute(1, 2, 0)
        # print('3333333333333',im_mat.shape)
        self.image = im_mat
        # self.save_as_pseudo_color_image()
        
        gt_mat = loadmat(self.gt_mat_path)
        self.mask = torch.from_numpy(gt_mat['Houston_gt']).long()  # mat's keys
        
        print('hsi data load',self.image.shape)
        print('hsi mask load',self.mask.shape)
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions do not match.")
    
    def _split_data(self):
        self.num_classes = len(np.unique(self.mask)) - (1 if 0 in np.unique(self.mask) else 0)  # 假设0为背景
        self.train_indicator, self.test_indicator = fixed_num_sample(
            self.mask, self.num_train_samples_per_class, self.num_classes, self.seed
        )
    
    def _pad_data(self):
        # 将图像和掩码转换为 [C, H, W] 和 [1, H, W]
        print('pading maks',self.image.shape)
        # image_np = self.image.transpose(2, 0, 1)  # [C, H, W]
        image_np = self.image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        mask_np = self.mask[np.newaxis, :, :]  # [1, H, W]
        train_indicator_np = self.train_indicator[np.newaxis, :, :]  # [1, H, W]
        test_indicator_np = self.test_indicator[np.newaxis, :, :]  # [1, H, W]
        
        # 更新类属性，不执行填充
        self.padded_image = image_np  # [C, H, W]
        self.padded_mask = mask_np  # [1, H, W]
        if self.training:
            self.padded_train_indicator = train_indicator_np  # [1, H, W]
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
        else:
            self.padded_test_indicator = test_indicator_np  # [1, H, W]
    
    def resample_minibatch(self):
        """
        重新采样训练小批量
        """
        seed = self.rs.randint(0, 2**32 - 1)
        self.train_inds_list = minibatch_sample(
            self.padded_mask, self.padded_train_indicator, self.sub_minibatch,
            seed=seed
        )
    
    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1  # 测试集只有一个批次
        
        # if self.training:
        #     train_inds = self.train_inds_list[idx]
        #     return {
        #         'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
        #         'mask':  self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
        #         'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
        #     }
        # else:
        #     return {
        #         'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
        #         'mask': self.padded_mask.clone().detach().to(dtype=torch.long),  # [H, W]
        #         'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [H, W]
        #     }
    
    def __getitem__(self, idx):
        if self.training:
            train_inds = self.train_inds_list[idx]
            data = {'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                    'train_inds': torch.tensor(train_inds, dtype=torch.bool)  # [H, W]
                    }
            if self.labeled:
                data['mask'] = self.padded_mask.clone().detach().to(dtype=torch.long)  # [H, W]
            return data
        
        else:
            data = {'image': self.padded_image.clone().detach().to(dtype=torch.float32),  # [C, H, W]
                    'test_inds': torch.tensor(self.padded_test_indicator, dtype=torch.bool)  # [1, H, W]
                    }
            if self.labeled:
                data['mask'] = self.padded_mask.clone().detach().to(dtype=torch.long)  # [1, H, W]
            return data

    def save_as_pseudo_color_image(self):
        # 选择三个通道用于 RGB
        if self.image.shape[0] < 3:
            raise ValueError("Not enough channels in the image")
        # 归一化每个通道
        r = self.image[:, :, 10]
        g = self.image[:, :, 30]
        b = self.image[:, :, 50]
        r_normalized = (r - r.min()) / (r.max() - r.min())
        g_normalized = (g - g.min()) / (g.max() - g.min())
        b_normalized = (b - b.min()) / (b.max() - b.min())

        # 合并通道制作彩色图像
        rgb_image = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)
        
        # 使用 matplotlib 保存伪彩色图像
        plt.imsave('pseudo_color_image.png', rgb_image)
        print("Pseudo-color image saved as 'pseudo_color_image.png'")


            
def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, minibatch_size, seed):
    """
    Args:
        gt_mask: 2-D array of shape [H, W]
        train_indicator: 2-D array of shape [H, W]
        minibatch_size: int
        seed: int

    Returns:
        List of 2-D arrays indicating selected training indices for each minibatch
    """
    if gt_mask.shape != train_indicator.shape:
        raise ValueError(f"gt_mask shape {gt_mask.shape} and train_indicator shape {train_indicator.shape} do not match.")
    
    rs = np.random.RandomState(seed)
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = {}
    
    for cls in cls_list:
        if cls == 0:
            continue  # 假设类标签从1开始，0为背景
        train_inds_per_class = (gt_mask == cls) & (train_indicator == 1)
        inds = np.where(train_inds_per_class.ravel())[0]
        rs.shuffle(inds)
        inds_dict_per_class[cls] = inds
    
    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            left = cnt * minibatch_size
            if left >= len(inds):
                continue
            right = min((cnt + 1) * minibatch_size, len(inds))
            fetch_inds = inds[left:right]
            train_inds[fetch_inds] = 1
        cnt += 1
        if train_inds.sum() == 0:
            break
        train_inds_list.append(train_inds.reshape(train_indicator.shape))
    
    return train_inds_list

class MinibatchSampler(Sampler):
    def __init__(self, dataset: NewPaviaDataset, seed=2333):
        """
        Args:
            dataset: NewPaviaDataset instance
            seed: int, random seed
        """
        self.dataset = dataset
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
    
    def __iter__(self):
        if self.dataset.training:
            self.dataset.resample_minibatch()
            indices = list(range(len(self.dataset)))
            shuffled_indices = torch.randperm(len(indices), generator=self.generator).tolist()
            return iter(shuffled_indices)
        else:
            return iter([0])
    
    def __len__(self):
        return len(self.dataset)

import random
import torch

def extract_samples(mask, test_inds, num_samples_per_class=10):
    """
    从test中提取每类10个样本，生成新的mask和val_inds
    
    Args:
        mask (Tensor): 包含类别标签的张量，形状为 [1, 1, H, W]
        test_inds (Tensor): 标记标签位置的布尔张量，形状为 [1, 1, H, W]
        num_samples_per_class (int): 每类要提取的样本数量，默认为10
    
    Returns:
        new_mask (Tensor): 新的mask，只包含被选中的样本
        val_inds (Tensor): 新的验证集指示张量
    """
    # 获取类别标签（忽略背景类0）
    classes = torch.unique(mask)
    if 0 in classes:
        classes = classes[1:]  # 假设0是背景类，跳过
    
    # 初始化新的mask和验证集指示张量
    new_mask = torch.zeros_like(mask, dtype=torch.long)
    val_inds = torch.zeros_like(test_inds, dtype=torch.bool)


    for class_id in classes:
        
        class_positions = torch.where((mask == class_id) & test_inds)
        
        if len(class_positions[0]) < num_samples_per_class:
            print(f"类别 {class_id} 样本数量不足，跳过")
            continue
            
        # 随机选择10个样本
        selected_indices = random.sample(range(len(class_positions[0])), num_samples_per_class)
        
        # 更新新的mask和验证集指示张量
        for idx in selected_indices:
            y = class_positions[2][idx]
            x = class_positions[3][idx]
            new_mask[0, 0, y, x] = class_id
            val_inds[0, 0, y, x] = True
    
    return new_mask, val_inds
    
if __name__ == "__main__":

    import torch
    import random


    image_mat_path = '../dataset/HU/Houston_em.mat'
    gt_mat_path = '../dataset/HU/Houston_gt.mat'
    train_dataset = NewHUDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                                num_train_samples_per_class=10,sub_minibatch=10,divisor=16,seed=2333)
    test_dataset = NewHUDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)
    
    # image_mat_path = 'dataset/IP/IP_edata.mat'
    # gt_mat_path = 'dataset/IP/Indian_pines_gt.mat'
    # train_dataset = NewIPDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
    #                             num_train_samples_per_class=5,sub_minibatch=5,divisor=16,seed=2333)
    # test_dataset = NewIPDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)

    # 创建采样器
    train_sampler = MinibatchSampler(train_dataset, seed=2333)
    test_sampler = MinibatchSampler(test_dataset, seed=2333)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset,batch_size=1,sampler=train_sampler,num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=1,sampler=test_sampler,num_workers=4)
    from collections import defaultdict


    # Process test data to select samples
    for batch in test_loader:
        images = batch['image']
        masks = batch['mask']
        test_inds = batch['test_inds']  # Assuming this contains the indices
        
        new_mask, val_inds = extract_samples( masks,test_inds,num_samples_per_class=10)

        values, counts = np.unique(masks, return_counts=True)

# 输出结果
        print("Value counts:")
        for value, count in zip(values, counts):
            print(f"Value {value}: {count} times")

        

              
        non_zero_indices = torch.nonzero(val_inds, as_tuple=False)

        # 获取非零元素的值
        non_zero_values = new_mask[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]

        #打印非零元素的索引和值
        # for index, value in zip(non_zero_indices, non_zero_values):
        #     print(f"Index: {index.tolist()}, Value: {value.item()}")
            
            # 计算非零样本的总数
        non_zero_count = non_zero_indices.size(0)

#打印总数
        print(f"Total number of non-zero samples: {non_zero_count}")
#     i=0
#     for batch in train_loader:
#         images = batch['image']
#         masks = batch['mask']
#         train_inds = batch['train_inds']
        
#         print(f"Batch image shape: {images.shape}")
#         print(f"Batch mask shape: {masks.shape}")
#         print(f"Batch train indices shape: {train_inds.shape}")
#         i=i+1
#         print(i)
        
#         # 获取所有非零元素的索引
#         # non_zero_indices = torch.nonzero(train_inds, as_tuple=False)

#         # # 获取非零元素的值
#         # non_zero_values = train_inds[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]

#         # # 打印非零元素的索引和值
#         # # for index, value in zip(non_zero_indices, non_zero_values):
#         # #     print(f"Index: {index.tolist()}, Value: {value.item()}")
            
#         #     # 计算非零样本的总数
#         # non_zero_count = non_zero_indices.size(0)

# # 打印总数
#         # print(f"Total number of non-zero samples: {non_zero_count}")
        
#     i=0
#     for batch in test_loader:
#         images = batch['image']
#         masks = batch['mask']
#         train_inds = batch['test_inds']
#         print(f"Batch image shape: {images.shape}")
#         print(f"Batch mask shape: {masks.shape}")
#         print(f"Batch train indices shape: {train_inds.shape}")
#         i=i+1
#         print(i)

#         non_zero_indices = torch.nonzero(train_inds, as_tuple=False)

#         # 获取非零元素的值
#         non_zero_values = train_inds[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]

#         # 打印非零元素的索引和值
#         # for index, value in zip(non_zero_indices, non_zero_values):
#         #     print(f"Index: {index.tolist()}, Value: {value.item()}")
            
#             # 计算非零样本的总数
#         non_zero_count = non_zero_indices.size(0)

# # 打印总数
#         print(f"Total number of non-zero samples: {non_zero_count}")
#         # print("Test batch size:", len(batch))
#         # break  # 只打印第一个批次
#         if non_zero_count > 0:
#             selected_class_indices = random.sample(non_zero_indices[:, 0].tolist(), min(sample_size_per_class, non_zero_count))
            
#             # 提取选中的样本
#             selected_samples = masks.view(-1)[selected_class_indices]
            
#             # 将选中的样本添加到训练集中
#             selected_train_data.append(selected_samples)
            
#             # 为选中的样本生成掩码
#             selected_mask = torch.zeros_like(masks.view(-1))
#             selected_mask[selected_class_indices] = 1  # 选中的样本对应掩码值为 1
            
#             # 恢复掩码到原始形状并存储
#             selected_mask_data.append(selected_mask.view(masks.shape))

#     # 打印每个类别选取的样本数量
#     for class_idx, count in class_samples_count.items():
#         print(f"Class {class_idx} selected samples: {count}")