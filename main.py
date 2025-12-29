import numpy as np

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as sio
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from utils.data_load import*
from utils.Udata_load import*
from utils.utils import*
from utils.datacolor import*


from model import*
import time
import os

import spectral

import argparse
import sys

from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser(description="Hyperspectral Classification")
parser.add_argument('--dataset', type=str, default='HU', choices=['samson','DC','IP','HU','PU','BE','LK'],
                            help="choose dataset: samson, DC, IP, HU")
parser.add_argument('--net', type=str, default='MSUANet', choices=['MSUANet','model1'],
                            help="choose network model: MSUANet, model1")
parser.add_argument('--train_num', type=int, default=20,help="number of training samples")
parser.add_argument('--epoch', type=int, default=300,help="number of training epochs")
parser.add_argument('--seed', type=int, default=2333, help="random seed")
args = parser.parse_args()

USE_TENSORBOARD = False  # whether to use tensorboard

set_seed(args.seed)
# Hyperparameters
NUM_EXPERIMENTS = 10
DATASET = args.dataset
TRAIN_NUM = args.train_num
EPOCH = args.epoch
MODEL = args.net

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

FILE_NAME = f"====Experiment====== {TRAIN_NUM}+{DATASET}+{EPOCH}+{MODEL}+ +{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

# visiualization and output directories
outimg_dir = f"output/{DATASET}/vis"
if not os.path.exists(outimg_dir):
    os.makedirs(outimg_dir)
out_vis_all = os.path.join(outimg_dir, f"{DATASET}_hsi_all.png")
out_vis_sel = os.path.join(outimg_dir, f"{DATASET}_hsi_sel.png")
abu_vis = os.path.join(outimg_dir, f"{DATASET}_abu_vis.png")
outtxt_dir = f"output/{DATASET}/txt"
if not os.path.exists(outtxt_dir):
    os.makedirs(outtxt_dir)    
outtxt_file = os.path.join(outtxt_dir, FILE_NAME)
data_color = get_data_color(DATASET)

best_folder = f"output/{DATASET}/{MODEL}/best_models{time.strftime('%Y-%m-%d_%H-%M-%S')}"
if not os.path.exists(best_folder):
    os.makedirs(best_folder)


dataset = DATASET

if dataset == 'DC':
    image_mat_path = 'dataset/DC/dc_um.mat'
    gt_mat_path = 'dataset/DC/DC_gt.mat'
    P, L, col,row  = 7, 191, 1280,307
    LR, EPOCH, batch_size = 1e-3, EPOCH, 1
    beta, delta, gamma = 0.5, 1e-2, 1e-7
    sparse_decay, weight_decay_param = 1e-6, 1e-4
    index = [0,1,2,3,4,5,6]
    config = {'in_channels': 191,'num_classes': 7,'block_channels': [64, 128, 256, 512],
              'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}
    
    train_dataset = NewDCDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                                num_train_samples_per_class=TRAIN_NUM,sub_minibatch=TRAIN_NUM,divisor=16,seed=2333)
    test_dataset = NewDCDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)

elif dataset == 'IP':
    image_mat_path = 'dataset/IP/IP_edata.mat'
    gt_mat_path = 'dataset/IP/Indian_pines_gt.mat'
    P, L, col,row  = 16, 200, 145,145
    LR, EPOCH, batch_size = 1e-3, EPOCH, 1
    beta, delta, gamma = 0.5, 1e-2, 1e-7
    sparse_decay, weight_decay_param = 1e-6, 1e-4
    index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    config = {'in_channels': 200,'num_classes': 16,'block_channels': [64, 128, 256, 512],
              'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}
    
    train_dataset = NewIPDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                            num_train_samples_per_class=TRAIN_NUM,sub_minibatch=TRAIN_NUM,divisor=16,seed=2333)
    test_dataset = NewIPDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)

elif dataset == 'HU':
    image_mat_path = 'dataset/HU/Houston_em.mat'
    gt_mat_path = 'dataset/HU/Houston_gt.mat'
    P, L, col,row  = 15, 144, 349,1905
    LR, EPOCH, batch_size = 1e-3, EPOCH, 1
    beta, delta, gamma = 0.5, 1e-2, 1e-7
    sparse_decay, weight_decay_param = 1e-6, 1e-4
    index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    config = {'in_channels': 144,'num_classes': 15,'block_channels': [64, 128, 256, 512],
              'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}
    train_dataset = NewHUDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                                num_train_samples_per_class=TRAIN_NUM,sub_minibatch=TRAIN_NUM,divisor=16,seed=2333)
    test_dataset = NewHUDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)

elif dataset == 'PU':

    image_mat_path = 'dataset/PU/PaviaU_edata.mat'
    gt_mat_path = 'dataset/PU/PaviaU_gt.mat'
    P, L, col,row  = 9, 103,610,340
    LR, EPOCH, batch_size = 1e-3, EPOCH, 1
    beta, delta, gamma = 0.5, 1e-2, 1e-7
    sparse_decay, weight_decay_param = 1e-6, 1e-4
    index = [0,1,2,3,4,5,6,7,8]
    config = {'in_channels': 103,'num_classes': 9,'block_channels': [64, 128, 256, 512],
            'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}
    
    train_dataset = NewPUDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                                num_train_samples_per_class=TRAIN_NUM,sub_minibatch=TRAIN_NUM,divisor=16,seed=2333)
    test_dataset = NewPUDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)

elif dataset == 'BE':
    
    image_mat_path = 'dataset/BE/Berlin_edata.mat'
    gt_mat_path = 'dataset/BE/Berlin_gt.mat'
    P, L, col,row  = 8, 244,1723,476
    LR, EPOCH, batch_size = 1e-3, EPOCH, 1
    beta, delta, gamma = 0.5, 1e-2, 1e-7
    sparse_decay, weight_decay_param = 1e-6, 1e-4
    index = [0,1,2,3,4,5,6,7]
    config = {'in_channels': 244,'num_classes': 8,'block_channels': [64, 128, 256, 512],
            'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}
    
    train_dataset = NewBEDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                                num_train_samples_per_class=TRAIN_NUM,sub_minibatch=TRAIN_NUM,divisor=16,seed=2333)
    test_dataset = NewBEDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)

else:
    raise ValueError("Unknown dataset")

data = sio.loadmat(image_mat_path)
Y = torch.from_numpy(data['Y'].astype(np.float32))
A = torch.from_numpy(data['A'])

print('===hsi data load===',Y.shape)
print('===abu data load===',A.shape)

M_true = data['M']  #normalized endmember matrix
M_true = M_true.astype(np.float32)
E_VCA_init = torch.from_numpy(data['M1']).unsqueeze(2).unsqueeze(3).float() # Init Endmember by VCA


Y = Y.reshape(L, row, col).permute(0, 2, 1)    # use permute to simulate column-major order
A = A.reshape(P, row, col).permute(0, 2, 1)    # use permute to simulate column-major order


HU_train_dataset= MyTrainData(img=Y,gt=A, transform=transforms.ToTensor())
HU_train_loader = torch.utils.data.DataLoader(dataset=HU_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


# Sample
train_sampler = MinibatchSampler(train_dataset, seed=2333)
test_sampler = MinibatchSampler(test_dataset, seed=2333)
# DataLoader
train_loader = DataLoader(train_dataset,batch_size=1, sampler=train_sampler,num_workers=4)
test_loader = DataLoader(test_dataset,batch_size=1,sampler=test_sampler,num_workers=4)

num_train_batches = len(train_loader)
num_hu_batches = len(HU_train_loader)
num_test_batches = len(test_loader) if 'test_loader' in locals() else 0

print(f"train data batch nums: {num_train_batches}")
print(f"HU train data batch nums: {num_hu_batches}")
print(f"test data batch nums: {num_test_batches}")

actual_iterations = min(len(train_loader), len(HU_train_loader))
print(f"actual training iterations/epoch: {actual_iterations}")

# run
oa_list = []
aa_list = []
kappa_list = []
class_acc_list = []
train_time_list = []
test_time_list = []

model_name = f"{MODEL}_test"  

for experiment in range(NUM_EXPERIMENTS):
    print(f"========= Experiment {experiment} ==========")
    if USE_TENSORBOARD:
        log_dir = os.path.join(
        "runs",
        DATASET,
        model_name,
        f"{experiment}")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志将保存到: {log_dir}")
    
    if MODEL == 'MSUANet': 
        from model import MSUANet
        net = MSUANet(num_classes=P,in_channel=L).to(device)

    else :
        print('no network can found')
    
    # loss
    HU_criterionSumToOne = SumToOneLoss().to(device)
    HU_criterionSparse = SparseKLloss()
    HU_re_loss_func = nn.MSELoss(size_average=True,reduce=True,reduction='mean')
    HC_criterion = nn.CrossEntropyLoss()

    model_dict = net.state_dict()
    model_dict['decoder1.0.weight'] = E_VCA_init
    net.load_state_dict(model_dict)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    dynamic_loss_module = DynamicWeightedLoss(num_losses=5).to(device)
    optimizer_loss_module = torch.optim.Adam(dynamic_loss_module.parameters(), lr=1e-3)

    #train    
    Total_loss = 0
    best_OA = 0
    best_epoch = 0
    for inputs_hu, labels_hu in HU_train_loader:
        abu_label = labels_hu.to(device)

    for batch in train_loader:
            images_train = batch['image'].to(device)
            masks = batch['mask'].squeeze(1).to(device)
            train_inds = batch['train_inds'].to(device)
    
    for idx in test_loader:
            images_eval = idx['image'].to(device)
            test_masks = idx['mask']        # 形状 (batch_size, 1, 1280, 307)
            test_inds = idx['test_inds']    # 形状 (batch_size, 1, 1280, 307)，布尔值

    time_train_start = time.time()
    for epoch in range(EPOCH):
        net.train()
        
        outputs_labeled, abu_pre, re_hsi_labeled  = net(images_train)

        selected_outputs = outputs_labeled.permute(0, 2, 3, 1)[train_inds.squeeze(1)]
        selected_masks = masks[train_inds.squeeze(1)].long() - 1
        HC_loss = HC_criterion(selected_outputs, selected_masks)


        loss_sumtoone = HU_criterionSumToOne(abu_pre)*1e-2
        loss_sparse = HU_criterionSparse(abu_pre)*1e-2
        loss_re = HU_re_loss_func(re_hsi_labeled, images_train)*1e-8
        loss_abu = HU_re_loss_func(abu_pre, abu_label) *1
        

        total_loss_HU = loss_sumtoone + loss_sparse + loss_re + loss_abu

        Total_loss = HC_loss + total_loss_HU 
        loss_list = [HC_loss, loss_sumtoone, loss_sparse, loss_re, loss_abu]
        total_loss = dynamic_loss_module(loss_list)

        optimizer.zero_grad()
        optimizer_loss_module.zero_grad()
        total_loss.backward()
        
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1) 
        optimizer.step()
        optimizer_loss_module.step()
        scheduler.step()


        net.eval()
        with torch.no_grad():
            outputs_val,_,_ = net(images_eval) 

            selected_val_outputs = outputs_val.permute(0, 2, 3, 1)[test_inds.squeeze(1)]
            test_masks1  = test_masks.squeeze(1).to(device)
            selected_val_masks = test_masks1[test_inds.squeeze(1)].long() - 1
            Te_loss = HC_criterion(selected_val_outputs, selected_val_masks)
            
            outputs_val = np.argmax(outputs_val.detach().cpu().numpy(), axis=1) + 1
            outputs_val = torch.from_numpy(outputs_val)                                
            y_pred = torch.masked_select(outputs_val.view(-1), test_inds.view(-1)) 
            val_masks = torch.masked_select(test_masks.view(-1), test_inds.view(-1)) 

            conf_matrix = confusion_matrix(val_masks, y_pred)
            class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            OA = np.trace(conf_matrix) / np.sum(conf_matrix)
            AA = np.mean(class_accuracy)
            total_pixels = np.sum(conf_matrix)
            pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_pixels ** 2)
            kappa = (OA - pe) / (1 - pe)

            if OA > best_OA:
                best_OA = OA
                best_epoch = epoch
                best_model_state = copy.deepcopy(net.state_dict())
                best_model_path = os.path.join(best_folder, f"best_model_epoch{experiment}.pth")
                torch.save(best_model_state, best_model_path)
                print(f"val oa: Epoch {best_epoch}, OA={OA:.4f} save in {best_model_path}")
                if USE_TENSORBOARD:
                    for i, w in enumerate(net.decoder_two.CSF_cls.shapley_weight):
                        writer.add_scalar(f'bestcls/fea{i}', w.item(), epoch)
                    for i, w in enumerate(net.decoder_two.CSF_um.shapley_weight):
                        writer.add_scalar(f'bestum/fea{i}', w.item(), epoch)
                    writer.add_histogram('bestcls/hit', net.decoder_two.CSF_cls.shapley_weight, epoch)
                    writer.add_histogram('bestum/hit', net.decoder_two.CSF_um.shapley_weight, epoch)

        if USE_TENSORBOARD:
            writer.add_scalar(f"Loss/Loss_total", Total_loss.item(), epoch)
            writer.add_scalar(f"Loss/Loss_HC", HC_loss.item(), epoch)
            writer.add_scalar(f"Loss/Loss_HU", total_loss_HU.item(), epoch)
            writer.add_scalar(f"Loss/Loss_abu", loss_abu.item(), epoch)
            writer.add_scalar(f"Loss/Loss_sparse", loss_sparse.item(), epoch)
            writer.add_scalar(f"Loss/Loss_re", loss_re.item(), epoch)
            writer.add_scalar(f"Loss/Loss_sumToOne", loss_sumtoone.item(), epoch)
            writer.add_scalar(f"Loss/Val_loss", Te_loss.item(), epoch)
            writer.add_scalar(f"Metrics/OA", OA.item(), epoch)
            writer.add_scalar(f"Metrics/AA", AA.item(), epoch)
            writer.add_scalar(f"Metrics/Kappa", kappa.item(), epoch)
            for i, w in enumerate(net.decoder_two.CSF_cls.shapley_weight):
                writer.add_scalar(f'cls/fea{i}', w.item(), epoch)
            for i, w in enumerate(net.decoder_two.CSF_um.shapley_weight):
                writer.add_scalar(f'um/fea{i}', w.item(), epoch)
            writer.add_histogram('cls/hit', net.decoder_two.CSF_cls.shapley_weight, epoch)
            writer.add_histogram('um/hit', net.decoder_two.CSF_um.shapley_weight, epoch)


            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar(f"Loss/Learning_Rate", current_lr, epoch)
            for name, param in net.named_parameters():
                writer.add_scalar(f"ParamsMean/{name}", param.data.mean().item(), epoch)
                writer.add_scalar(f"ParamsStd/{name}", param.data.std().item(), epoch)

            if epoch % 10 == 0:
                for name, param in net.named_parameters():
                    writer.add_histogram(f"ParamsDist/{name}", param.detach().cpu(), epoch)

        if epoch % 1 == 0:
            print(f"Epoch: {epoch+1}: "
                f"TLoss: {Total_loss.item():.4f}, "
                f"HC: {HC_loss.item():.4f}, "
                f"HU: {total_loss_HU.item():.4f}, "
                f"abu: {loss_abu.item():.4f}, "
                f"Re: {loss_re.item():.4f}, "
                f"Sto: {loss_sumtoone.item():.4f}, "
                f"Spa: {loss_sparse.item():.4f}, "
                f"oa: {OA.item():.4f}, "
                f"aa: {AA.item():.4f}, "
                f"kappa: {kappa.item():.4f}, "
                f"Val_loss: {Te_loss.item():.4f} "
                )

                
    print('Finished Training')
    time_train_end = time.time()
    time_test_start = time.time()

    net.load_state_dict(torch.load(best_model_path))
    net.eval()

    with torch.no_grad():

        for idx in test_loader:

            images = idx['image'].to(device)
            masks = idx['mask']
            test_inds = idx['test_inds'] #1, 1, 1280, 307
            outputs,abu_est,_ = net(images) # 1, 7, 1280, 307

            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1) + 1
            
            output_image = outputs.squeeze(0) 
            train_inds = train_inds.to('cpu')

            testmask = (test_inds + train_inds ).cpu().numpy().astype(np.float32)
            testmask = testmask.squeeze(0).squeeze(0) 
            
            spectral.save_rgb(out_vis_all,output_image,colors = data_color)
            spectral.save_rgb(out_vis_sel,output_image*testmask,colors = data_color)

            print(f"Saved: {out_vis_sel}")
            outputs = torch.from_numpy(outputs)                                # 1, 1280, 307
            masks = torch.masked_select(masks.view(-1), test_inds.view(-1))    # 7729
            
            y_pred = torch.masked_select(outputs.view(-1), test_inds.view(-1)) # 7729
            # HU
            abu_est = abu_est/(torch.sum(abu_est, dim=1, keepdim=True))
            abu_est = torch.clamp(abu_est, min=0.0, max=1.0)
            abu_est = torch.reshape(abu_est.squeeze(0), (P, col, row)).cpu().detach().numpy()
            
        conf_matrix = confusion_matrix(masks, y_pred)
        print(conf_matrix)
        report = classification_report(masks, y_pred, digits=4)
        print("Classification Report:\n")
        print(report)

        class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
        class_acc_list.append(class_accuracy)  
        
        OA = np.trace(conf_matrix) / np.sum(conf_matrix)
        oa_list.append(OA)
        
        AA = np.mean(class_accuracy)
        aa_list.append(AA)
        
        total_pixels = np.sum(conf_matrix)
        pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_pixels ** 2)
        kappa = (OA - pe) / (1 - pe)
        kappa_list.append(kappa)
        
        iou = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))
        MIoU = np.mean(iou)
        
        print(f"Overall Accuracy (OA): {OA}")
        print(f"Average Accuracy (AA): {AA}")
        print(f"Kappa: {kappa}")
        print(f"Mean Intersection over Union (MIoU): {MIoU}")
        
        B = A[index,:,:]
        B = B.detach().numpy()
        
        # Y = Y.detach().numpy()
        sio.savemat('A_matrix.mat', {'A': abu_est})
        print('**********************************')
        print('RMSE: {:.5f}'.format(compute_rmse(A, abu_est)))
        
        plt.figure(figsize=(15, 10), constrained_layout=True)
        for i in range(P):
            plt.subplot(2, P, i+1)
            plt.imshow(abu_est[i, :, :].T)
            plt.axis('off')  
        for i in range(P):
            plt.subplot(2, P, P+i+1)
            plt.imshow(A[i, :, :].T)
            plt.axis('off')  
        
        plt.savefig(abu_vis, bbox_inches='tight', pad_inches=0.1)
                       
    time_test_end = time.time()
    print('train computational cost:', time_train_end-time_train_start)
    print('test computational cost:',  time_test_end-time_test_start)
    print('**********************************')
    train_time_list.append(time_train_end-time_train_start)
    test_time_list.append(time_test_end-time_test_start)


    with open(outtxt_file, "a") as f:
        
            f.write("Overall Accuracy (OA): {:.4f}\n".format(OA))
            f.write("Average Accuracy (AA): {:.4f}\n".format(AA))
            f.write("Kappa Coefficient: {:.4f}\n".format(kappa))
            for cls_idx, acc in enumerate(class_accuracy, 1):
                f.write(f"Class {cls_idx}: {acc:.4f}\n")
            f.write('Train computational cost: {:.4f} \n'.format(time_train_end - time_train_start))
            f.write('Test computational cost: {:.4f} \n'.format(time_test_end - time_test_start))
            f.write('**********************************\n')

    torch.cuda.empty_cache()
if class_acc_list:
        
        class_acc_matrix = np.vstack(class_acc_list)
        class_avg = np.nanmean(class_acc_matrix, axis=0)
        class_std = np.nanstd(class_acc_matrix, axis=0)
        
        with open(outtxt_file, "a") as f:
            f.write("\n=== Final Statistics ===\n")
            f.write(f"Average OA: {np.mean(oa_list):.4f} ± {np.std(oa_list):.4f}\n")
            f.write(f"Average AA: {np.mean(aa_list):.4f} ± {np.std(aa_list):.4f}\n")
            f.write(f"Average Kappa: {np.mean(kappa_list):.4f} ± {np.std(kappa_list):.4f}\n")
            f.write("\nPer-Class Accuracy (Mean ± Std):\n")
            for cls_idx, (avg, std) in enumerate(zip(class_avg, class_std), 1):
                f.write(f"Class {cls_idx}: {avg:.4f} ± {std:.4f}\n")
            f.write(f"Average Train time: {np.mean(train_time_list):.4f} ± {np.std(train_time_list):.4f}\n")
            f.write(f"Average test time: {np.mean(test_time_list):.4f} ± {np.std(test_time_list):.4f}\n")
if USE_TENSORBOARD:
    writer.close()
    print("全部实验结束，TensorBoard 日志已记录完毕。")
    print("如需查看，请在终端执行:  tensorboard --logdir=runs")
        