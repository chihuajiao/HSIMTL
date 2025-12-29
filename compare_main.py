import numpy as np

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as sio
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
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
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():

    parser = argparse.ArgumentParser(description="Hyperspectral Classification")
    parser.add_argument('--dataset', type=str, default='BE', choices=['DC','IP','HU','BE'],
                            help="choose dataset:  DC, IP, HU ,BE")
    parser.add_argument('--net', type=str, default='TransUNet', choices=['UNet','FreeNet','SSFCN','FContNet',
                                                                       'UperNet','Segformer','TransUNet'],
                            help="choose network model: UNet, FreeNet, SSFCN, FContNet, UperNet, Segformer, TransUNet")
    parser.add_argument('--train_num', type=int, default=1,help="training samples per class")
    parser.add_argument('--seed', type=int, default=2333, help="random seed")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID to use")
    parser.add_argument('--epoch', type=int, default=300,help="number of training epochs")

    args = parser.parse_args()

    set_seed(args.seed)

    # Hyperparameters         
    NUM_EXPERIMENTS = 10    
    DATASET = args.dataset       
    MODEL   = args.net
    TRAIN_NUM   = args.train_num
    EPOCH   = args.epoch

    USE_TENSORBOARD = False  # whether to use tensorboard
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    FILE_NAME = f"=======Comparison Experiment {EPOCH} {TRAIN_NUM}+{DATASET}_{MODEL} {time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    # Visualization path
    outimg_dir = f"compare_output/{DATASET}/vis"
    if not os.path.exists(outimg_dir):
        os.makedirs(outimg_dir)
    out_vis_all = os.path.join(outimg_dir, f"{MODEL}_{TRAIN_NUM}hsi_all.png")
    out_vis_sel = os.path.join(outimg_dir, f"{MODEL}_{TRAIN_NUM}hsi_sel.png")
    outtxt_dir = f"compare_output/{DATASET}/txt"
    if not os.path.exists(outtxt_dir):
        os.makedirs(outtxt_dir)    
    outtxt_file = os.path.join(outtxt_dir, FILE_NAME)
    data_color = get_data_color(DATASET)
    best_folder = f"compare_output/{DATASET}/{MODEL}/best_models{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(best_folder):
        os.makedirs(best_folder)

    dataset = DATASET
    if dataset == 'samson': # This is a test
        image_file = r'dataset/HSIU/samson_dataset.mat'
        P, L, col,row = 3, 156, 95,95
        LR, EPOCH, batch_size = 2e-3, 1, 1
        beta, delta, gamma = 0.5, 1e-3, 1e-7
        sparse_decay, weight_decay_param = 5e-6, 1e-4
        index = [1,2,0]

    elif dataset == 'DC':
        image_mat_path = 'dataset/DC/dc_um.mat'
        gt_mat_path = 'dataset/DC/DC_gt.mat'
        P, L, col,row  = 7, 191, 1280,307
        LR, EPOCH, batch_size = 1e-3, 300, 1
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
        LR, EPOCH, batch_size = 1e-3, 300, 1
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
        LR, EPOCH, batch_size = 1e-3, 300, 1
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
        P, L, col,row  = 9, 103,340,610
        LR, EPOCH, batch_size = 1e-3, 300, 1
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
        P, L, col,row  = 8, 244,476,1723
        LR, EPOCH, batch_size = 1e-3, 300, 1
        beta, delta, gamma = 0.5, 1e-2, 1e-7
        sparse_decay, weight_decay_param = 1e-6, 1e-4
        index = [0,1,2,3,4,5,6,7]
        config = {'in_channels': 244,'num_classes': 8,'block_channels': [64, 128, 256, 512],
                'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}
        train_dataset = NewBEDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                                    num_train_samples_per_class=TRAIN_NUM,sub_minibatch=TRAIN_NUM,divisor=16,seed=2333)
        test_dataset = NewBEDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)
    elif dataset == 'LK':
    
        image_mat_path = 'dataset/LK/LongKou_edata.mat'
        gt_mat_path = 'dataset/LK/WHU_Hi_LongKou_gt.mat'
        P, L, col,row  = 9, 270,400,550
        LR, EPOCH, batch_size = 1e-3, 300, 1
        beta, delta, gamma = 0.5, 1e-2, 1e-7
        sparse_decay, weight_decay_param = 1e-6, 1e-4
        index = [0,1,2,3,4,5,6,7]
        config = {'in_channels': 270,'num_classes': 9,'block_channels': [64, 128, 256, 512],
                'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}

        train_dataset = NewLKDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=True,
                                num_train_samples_per_class=TRAIN_NUM,sub_minibatch=TRAIN_NUM,divisor=16,seed=2333)
        test_dataset = NewLKDataset(image_mat_path=image_mat_path,gt_mat_path=gt_mat_path,training=False,divisor=16,seed=2333)

    else:
        raise ValueError("Unknown dataset")


    # Sample
    train_sampler = MinibatchSampler(train_dataset, seed=2333)
    test_sampler = MinibatchSampler(test_dataset, seed=2333)
    # DataLoader
    train_loader = DataLoader(train_dataset,batch_size=1, sampler=train_sampler,num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=1,sampler=test_sampler,num_workers=4)

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
            print(f"TensorBoard logs will be saved to: {log_dir}")

        if MODEL == 'UNet': 
            from comparemethod.UNet import UNet
            net = UNet(n_channels=config['in_channels'], n_classes=config['num_classes']).to(device)
        elif MODEL == 'FreeNet':
            from comparemethod.FreeNet import FreeNet
            free_con = {'in_channels': config['in_channels'],'num_classes': config['num_classes'],
                        'block_channels': [64, 128, 256, 512],'num_blocks': [2, 2, 2, 2],'inner_dim': 128,'reduction_ratio': 0.5}
            net = FreeNet(free_con).to(device)
        elif MODEL == 'FContNet':
            from comparemethod.FullyContNet import fucontnet

            args.ma = 1
            args.mi = -1
            args.norm = 'std'
            args.network = 'FContNet'
            args.head = 'psp'
            args.mode = 'p_s_c'
            args.input_size = [col,row]
            print('Implementing FcontNet in {} mode with {} head!'.format(args.mode,args.head))
            net = fucontnet(args, config['in_channels'],  config['num_classes']).to(device)
        elif MODEL == 'UperNet':
            from comparemethod.UperNet import UPerNet
            net = UPerNet(num_classes=config['num_classes'],firstchannel=config['in_channels']).to(device) 
        elif MODEL == 'SSFCN':
            from comparemethod.SSFCN import SSFCN
            net = SSFCN(num_bands=config['in_channels'],num_classes=config['num_classes']).to(device) 
    
        elif MODEL == 'Segformer':
            from comparemethod.Segformer import Segformer
            # use the MiT-B0
            net = Segformer(dims = (32, 64, 160, 256),      
                            heads = (1, 2, 5, 8),           
                            ff_expansion = (8, 8, 4, 4),    
                            reduction_ratio = (8, 4, 2, 1), 
                            num_layers = 2,                 
                            decoder_dim = 256,              
                            num_classes = config['num_classes'],                 
                            channels= config['in_channels']).to(device) 
        elif MODEL == 'TransUNet':
            from comparemethod.TransUNet import VisionTransformer,CONFIGS
            config_vit = CONFIGS["R50-ViT-B_16"]# R50-ViT-B_16
            h_new = ((col + 15) // 16) * 16
            w_new = ((row + 15) // 16) * 16
            img_size = (h_new,w_new)
            config_vit.n_classes = config['num_classes']
            config_vit.inchannel = config['in_channels']
            config_vit.patches.grid = (int(img_size[0] / 16), int(img_size[1] / 16))
            config_vit.num_patches_h = int(img_size[0] / 16)
            config_vit.num_patches_w = int(img_size[1] / 16)
            net = VisionTransformer(config_vit, img_size=img_size).to(device)
            LR =0.01 
            
        else:
            print('no network can found')

        for batch in train_loader:
            images_train = batch['image'].to(device)
            masks = batch['mask'].squeeze(1).to(device)
            train_inds = batch['train_inds'].to(device)
    
        for idx in test_loader:
            images_eval = idx['image'].to(device)
            test_masks = idx['mask'] 
            test_inds = idx['test_inds'] 

        
        
        HC_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        Total_loss = 0
        best_OA = 0
        best_epoch = 0


        #train
        Total_loss = 0
        time_train_start = time.time()
        for epoch in range(EPOCH):
            
            net.train()
            outputs_labeled = net(images_train)

            selected_outputs = outputs_labeled.permute(0, 2, 3, 1)[train_inds.squeeze(1)]
            selected_masks = masks[train_inds.squeeze(1)].long() - 1
            HC_loss = HC_criterion(selected_outputs, selected_masks)

            Total_loss = HC_loss

            #backward
            optimizer.zero_grad()
            Total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1) 
            optimizer.step()
            scheduler.step()

            net.eval()
            with torch.no_grad():
                outputs_val = net(images_eval) # 1, 7, 1280, 307
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
                print(f"val oa: Epoch {best_epoch}, OA={OA:.4f} saved to {best_model_path}")

                
            if epoch % 1 == 0:
                print(f"Epoch: {epoch+1}: "
                    f"TLoss: {Total_loss.item():.4f}, "
                    f"HC: {HC_loss.item():.4f}, "
                    f"oa: {OA.item():.4f}, "
                    f"aa: {AA.item():.4f}, "
                    f"kappa: {kappa.item():.4f}, "
                    f"Val_Loss: {Te_loss.item():.4f} "
                    )        

            if USE_TENSORBOARD:
                writer.add_scalar(f"Loss/Loss_total", Total_loss.item(), epoch)
                writer.add_scalar(f"Loss/Loss_HC", HC_loss.item(), epoch)
                writer.add_scalar(f"Loss/Val_loss", Te_loss.item(), epoch)
                writer.add_scalar(f"Metrics/OA", OA.item(), epoch)
                writer.add_scalar(f"Metrics/AA", AA.item(), epoch)
                writer.add_scalar(f"Metrics/Kappa", kappa.item(), epoch)

                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar(f"Loss/Learning_Rate", current_lr, epoch)
                for name, param in net.named_parameters():
                    writer.add_scalar(f"ParamsMean/{name}", param.data.mean().item(), epoch)
                    writer.add_scalar(f"ParamsStd/{name}", param.data.std().item(), epoch)

                if epoch % 10 == 0:
                    for name, param in net.named_parameters():
                        writer.add_histogram(f"ParamsDist/{name}", param.detach().cpu(), epoch)        
        print('Finished Training')
        time_train_end = time.time()
        time_test_start = time.time()
        
        #test
        #HC
        net.load_state_dict(torch.load(best_model_path))
        net.eval()
        with torch.no_grad():
            
            for idx in test_loader:

                images = idx['image']
                masks = idx['mask']
                test_inds = idx['test_inds'] #1, 1, 1280, 307
                
                images = images.to(device)
                outputs= net(images) # 1, 7, 1280, 307

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
                
                
            conf_matrix = confusion_matrix(masks, y_pred)
            print(conf_matrix)
            report = classification_report(masks, y_pred, digits=4)
            print("Classification Report:\n")
            print(report)

            class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            class_acc_list.append(class_accuracy)  # 收集结果
            
            OA = np.trace(conf_matrix) / np.sum(conf_matrix)
            oa_list.append(OA)
            
            class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
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

if __name__ == "__main__":
    print("Command line arguments:", sys.argv)  # Print the passed arguments
    main()  