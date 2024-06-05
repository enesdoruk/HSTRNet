import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import wandb

from train_ifnet import ifnet
from model.ifnet.RIFE import Model
from train_hstrnet import hstrnet
from model.HSTR_RIFE_v5_scaled import HSTRNet




def args_config():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--bs_train", default=16, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--ifnet_load", default='./model_dict/ifnet.pkl', type=str)
    parser.add_argument("--contexnet_load", default=None, type=str)
    parser.add_argument("--unet_load", default=None, type=str)
    parser.add_argument("--optimizer_load", default=None, type=str)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--bs_val", default=4, type=int)
    parser.add_argument("--transform", default=False, type=bool)
    parser.add_argument("--data_parallel", default=False, type=bool)
    parser.add_argument("--finetune", default=False, type=bool)
    parser.add_argument("--hstrnet", default=True, type=bool)
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    args = parser.parse_args()

    return args



args = args_config()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.hstrnet:
        if args.data_parallel:
            model = nn.DataParallel(HSTRNet(device))
        else:
            model = HSTRNet(device)

        try:
            hstrnet(model, args.transform, args.epoch, args.bs_train, args.bs_val, args.lr, args.ifnet_load, \
                    args.contexnet_load, args.unet_load, args.optimizer_load,  args.workers, args.weight_decay, args.finetune)
        except Exception as e:
            logging.exception("Unexpected exception! %s", e)

    else:
        if args.data_parallel:
            model = nn.DataParallel(Model())
        else:
            model = Model()

        try:
            ifnet(model, args.epoch, args.lr, args.local_rank, args.bs_train, finetune=args.finetune, transform=False)
        except Exception as e:
            logging.exception("Unexpected exception! %s", e)


if args.finetune:
    sweep_configuration = { 'method': 'bayes',
                            'name': 'hyperparameter tuning',
                            'metric': {'goal': 'minimize', 'name': 'test/PSNR'},
                            'parameters': {'epochs': {'values': [5, 10, 15]},
                                           'lr': {'values': [1e-2, 1e-4, 1e-6, 1e-8]},
                                           'bs': {'values': [4,8,16]},
                                           }}

    sweep_id = wandb.sweep(sweep=sweep_configuration,  project='hstrnet-hyperparameter-tuning')

    wandb.agent(sweep_id, function=main, count=10)

else:
    main()