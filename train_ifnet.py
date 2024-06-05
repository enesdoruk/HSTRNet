import math
import torch
import numpy as np
import wandb

from model.ifnet.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader

device = torch.device("cuda")
log_path = 'train_log'
netron = True


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def ifnet(model, epoch, lr, local_rank, batch_size, finetune=False, transform=False):
    global netron  

    config = {
                "epoch": epoch, 
                "lr": lr, 
                "batch_size": batch_size
                }
    
    wandb.init(project="hstrnet", name='{}-{}-{}-{}'.format(epoch, batch_size, lr, transform), config=config)

    if finetune:
        if 'epoch' in wandb.config:
            epoch = wandb.config.epoch
        if 'lr' in wandb.config:
            lr = wandb.config.lr
        if 'batch_size' in wandb.config:
            batch_size = wandb.config.batch_size


    step = 0
    nr_eval = 0

    dataset = VimeoDataset("train", device, transform=transform)
    train_data = DataLoader(dataset, batch_size= batch_size, num_workers=8, pin_memory=True, drop_last=True)
    
    step_per_epoch = train_data.__len__()
   
    dataset_val = VimeoDataset("validation", device, transform=transform)
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
   
    print('training...')
    
    for epoch in range(epoch):  
        loss_l1 = 0     
        loss_tea = 0 
        loss_distill = 0
        for i, data in enumerate(train_data):
            data_gpu = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                     
            imgs = data_gpu[:, 3:9]
            gt = data_gpu[:, :3]
           
            learning_rate = lr
           
            pred, info = model.update(imgs, gt, learning_rate, training=True) # pass timestep if you are training RIFEm

            if netron:
                import cv2
                from torchviz import make_dot
                make_dot(pred.mean(), params=dict(model.named_parameters())).render("network_ifnet", format="png")

                network_img = cv2.imread("network_ifnet.png")
                arch = wandb.Image(network_img, caption="Network architecture")
                wandb.log({"network_architecture": arch})

                netron = False

                      
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()

            loss_l1 += info['loss_l1']
            loss_tea += info['loss_tea']
            loss_distill += info['loss_distill']
            
            step += 1

        print('epoch:{} {}/{}  loss:{:.4e}'.format(epoch, i, step_per_epoch, ((float(loss_l1) + float(loss_tea) + float(loss_distill))/int(i))/3))
        print('epoch:{} {}/{}  loss_l1:{:.4e}'.format(epoch, i, step_per_epoch, loss_l1/i))
        print('epoch:{} {}/{}  loss_tea:{:.4e}'.format(epoch, i, step_per_epoch, loss_tea/i))
        print('epoch:{} {}/{}  loss_distill:{:.4e}'.format(epoch, i, step_per_epoch, loss_distill/i))
        print('*'*50)
       
        nr_eval += 1
        
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, local_rank, epoch)
        model.save_model(log_path, local_rank)    

def evaluate(model, val_data, nr_eval, local_rank, epoch):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []

    for i, data in enumerate(val_data):
        data_gpu = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
        imgs = data_gpu[:, 3:9]
        gt = data_gpu[:, :3]
       
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
       
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
       
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
      
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
    

    print(f"epoch:{epoch} Val loss :  ", (float(loss_l1_list + loss_tea_list + loss_distill_list)/int(len(loss_l1_list))/3))
    print(f"epoch:{epoch} Val loss_l1 :  ", sum(loss_l1_list)/len(loss_l1_list))
    print(f"epoch:{epoch} Val loss_tea :  ", sum(loss_tea_list)/len(loss_tea_list))
    print(f"epoch:{epoch} Val loss_distill :  ", sum(loss_distill_list)/len(loss_distill_list))
    print('-'*50)
    
    if local_rank != 0:
        return