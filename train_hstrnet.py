import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
import wandb
import os

from tqdm import tqdm
from dataset import VimeoDataset, DataLoader, MAMIDataset
from model.pytorch_msssim import ssim_matlab
from CX.CSFlow import *



netron = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hstrnet(model, transform, epochs, bs_tr, bs_val, lr, ifnet_load, \
                contexnet_load, unet_load, optimizer_load, workers, weight_decay, finetune):
    
    global netron  

    config = {
                "epochs": epochs, 
                "learning_rate": lr, 
                "batch_size": bs_tr
                }
    
    wandb.init(project="HSTRNET_demo", name='context_loss_w_weight', config=config)

    if finetune:
        if 'epochs' in wandb.config:
            epochs = wandb.config.epochs
        if 'lr' in wandb.config:
            lr = wandb.config.lr
        if 'bs' in wandb.config:
            bs_tr = wandb.config.bs
    

    print("Device: ", device)

    #increased_train_dataset = torch.utils.data.ConcatDataset([dataset_train,dataset_train_aug])
    
    # dataset_train = VimeoDataset("train", device, transform=transform)
    # train_data = DataLoader(dataset_train, batch_size=bs_tr, num_workers=workers, drop_last=True, shuffle=True)
    dataset_train = MAMIDataset("train", f"{os.path.abspath(os.getcwd())}/dataset/MAMI/train", device)
    train_data = DataLoader(
        dataset_train, batch_size=bs_tr, num_workers=workers, drop_last=True, shuffle=True
    )
    
    print("Training dataset is loaded")

    # dataset_val = VimeoDataset("validation", device, transform=transform)
    # val_data = DataLoader(dataset_val, batch_size=bs_val, num_workers=workers, shuffle=False, drop_last=True)
    dataset_val = MAMIDataset("validation", f"{os.path.abspath(os.getcwd())}/dataset/MAMI/test", device)
    val_data = DataLoader(dataset_val, batch_size=bs_val, num_workers=workers, shuffle=False)


    print("Validation dataset is loaded")

    len_val = dataset_val.__len__()
    L1_lossFn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    print("Training...")
    print("Train data set size:" + str(train_data.__len__()))
    print("Val data set size:" + str(val_data.__len__()))
    print("Training is starting")


    if ifnet_load is not None:
        model.ifnet.load_state_dict(torch.load(ifnet_load, map_location=device))

        for k, v in model.ifnet.named_parameters():
            v.requires_grad = False
    
    if contexnet_load is not None:
        model.contextnet.load_state_dict(torch.load(contexnet_load, map_location=device))

        for k, v in model.contextnet.named_parameters():
            v.requires_grad = False

    if unet_load is not None:
        model.unet.load_state_dict(torch.load(unet_load, map_location=device))

        for k, v in model.unet.named_parameters():
            v.requires_grad = False
    
    if optimizer_load is not None:
        optimizer.load_state_dict(torch.load(optimizer_load, map_location=device))

        for k, v in optimizer.unet.named_parameters():
            v.requires_grad = False
    

    start = time.time()

    context_lambda = 0.1

    loss = 0
    psnr_list = []
    ssim_list = []
    for epoch in range(epochs): 
        #model.ifnet.train()
        model.contextnet.train()
        model.unet.train()
        loss = 0

        print("Epoch: ", epoch)
        print("---------------------------------------------")
        print("Epoch:" + str(epoch))
        print("---------------------------------------------")

        for trainIndex, data in enumerate(tqdm(train_data)):            
            #model.ifnet.train()
            model.contextnet.train()
            model.unet.train()            
            
            data = data.to(device, non_blocking=True) / 255.0
            data = data.float()
            
            gt = data[:, :3]
            ref = data[:, 3:6]
            lr = data[:, 6:9]
            
            """ image_show(gt)
            image_show(ref)
            image_show(lr) """

            
            imgs = torch.cat((ref, lr), 1)
            optimizer.zero_grad()
            pred = model(imgs)

            if netron:
                import cv2
                from torchviz import make_dot
                make_dot(pred.mean(), params=dict(model.named_parameters())).render("network_hstrnet", format="png")

                network_img = cv2.imread("network_hstrnet.png")
                arch = wandb.Image(network_img, caption="Network architecture")
                wandb.log({"network_architecture": arch})


                netron = False
            

            # wandb_pred = imgs.detach().to('cpu').numpy()

            # table = wandb.Table(columns=['ID', "Train pred image" ])
            # counter = 0
            # for imgs in wandb_pred:
            #     counter += 1
            #     pred_img = np.transpose(imgs, (1, 2, 0))
            #     table.add_data(counter, wandb.Image(pred_img[:,:,:3]))
                
            # wandb.log({"train/Table" : table})
            
          
            L1_loss = L1_lossFn(pred, gt)
            context_loss = CX_loss(pred.cpu().detach().numpy(), gt.cpu().detach().numpy(), distance=Distance.L2, nnsigma=float(1.0))
            L1_loss += context_loss.numpy() * context_lambda
            
            L1_loss.backward()
            optimizer.step()
            loss += float(L1_loss.item())
            end = time.time()


            if trainIndex == (train_data.__len__() - 1):
                print("Validating, Train Index: " + str(trainIndex))

                with torch.no_grad():
                    psnr, ssim = validate_hstrnet(model, val_data, len_val, 1)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    endVal = time.time()

                print(
                    " Loss: %0.6f  TrainExecTime: %0.1f  ValPSNR: %0.4f  ValEvalTime: %0.2f  SSIM: %0.4f "
                    % (loss / trainIndex, end - start, psnr, endVal - end, ssim)
                )

                wandb.log({'train/loss': round(loss / trainIndex, 6), "val/PSNR": round(psnr, 4), "val/SSIM": round(ssim, 4)})

                print(
                    "Train index: "
                    + str(trainIndex)
                    + " Loss: "
                    + str(round(loss / trainIndex, 6))
                    + " TrainExecTime: "
                    + str(round(end - start, 1))
                    + " ValPSNR: "
                    + str(round(psnr, 4))
                    + " ValEvalTime: "
                    + str(round(endVal - end, 2))
                    + " SSIM: "
                    + str(round(ssim, 4))
                )

                start = time.time()

        torch.save(model.contextnet.state_dict(), "model_dict/HSTR_contextnet_" + str(epoch) + ".pkl")
        torch.save(model.unet.state_dict(), "model_dict/HSTR_unet_" + str(epoch) + ".pkl")
        torch.save(optimizer.state_dict(), "model_dict/HSTR_optimizer" + str(epoch) + ".pkl")
        
    val_data_last = DataLoader(dataset_val, batch_size=bs_val, num_workers=workers, shuffle=False)

    psnr, ssim = validate_hstrnet(model, val_data_last, len_val, 1)
    print("------------------------------------------")
    print(
        "Last evaluation --> PSNR:"
        + str(psnr)
        + " SSIM:"
        + str(ssim)
    )
    print("PSNR:" + str(psnr) + " SSIM:" + str(ssim))

    wandb.log({"test/PSNR": psnr, "test/SSIM": ssim})


def validate_hstrnet(model, val_data, len_val, batch_size):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []

    for valIndex, data in enumerate(tqdm(val_data)):
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
    
            gt = data[:, :3]
            ref = data[:, 3:6]
            lr = data[:, 6:9]
            
            """image_show(gt)
            image_show(ref)
            image_show(lr)"""
            
            imgs = torch.cat((ref, lr), 1)
    
            pred = model(imgs)


            # wandb_pred = pred.detach().to('cpu').numpy()

            # table = wandb.Table(columns=['ID', "Val pred image" ])
            # counter = 0
            # for imgs in wandb_pred:
            #     counter += 1
            #     pred_img = np.transpose(imgs, (1, 2, 0))
            #     table.add_data(counter, wandb.Image(pred_img[:,:,:3]))
                
            # wandb.log({"val/Table" : table})

            for i in range(int(pred.shape[0])):
                if(((gt[i: i+1,:] - pred[i: i+1,:]) * (gt[i: i+1,:] - pred[i: i+1,:])).mean()) < 0:
                    continue
                psnr = -10 * math.log10(((gt[i: i+1,:] - pred[i: i+1,:]) * (gt[i: i+1,:] - pred[i: i+1,:])).mean())
                ssim_ = float(ssim_matlab(pred[i: i+1,:], gt[i: i+1,:]))
                psnr_list.append(psnr)
                ssim_list.append(ssim_)
    print("SSIM: ", np.mean(ssim_list), "   PSNR: ", np.mean(psnr_list))

    return np.mean(psnr_list), np.mean(ssim_list)



