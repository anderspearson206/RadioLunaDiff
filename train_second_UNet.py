# THIS CODE WAS MODIFEID FROM PMNET
# https://github.com/abman23/pmnet

from __future__ import print_function, division
import os
import time

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled
from k2net.diff_modules import K2_UNet
from dataset.refraction_loaders_mhz import RefractLunarLoader, RefractLunarLoader2

from tqdm import tqdm
from datetime import datetime
import sys

from torchsummary import summary as summary_

import argparse
import importlib
from pmnet.eval.eval import L1_loss, MSE, RMSE
import random
import cv2
import matplotlib.pyplot as plt
seed = 206
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from PIL import Image
# RESULT_FOLDER = '/content/drive/MyDrive/Colab Notebooks/Joohan/PMNet_Extension_Result'
RESULT_FOLDER = './results/train_pmnet'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'



################################################################################
# DATA LOADING
################################################################################
def load_lunar_for_hz(data_dir1, data_dir2):
    args = {
        'train_split_ratio': 0.7, 'val_split_ratio': 0.15,
        'use_los_input': False, 'use_pathloss_input': False,
        'non_global_norm': False,
        'filt_hm': False, 'use_filt_and_norm': True,
        'los_weight': .2, 'crop_resize': False, 'tx_array': False
    }
    num_maps = 500
    Radio_val1 = RefractLunarLoader(phase="val", root_dir=data_dir1, rm_path='rm58', freq = 1, k2=False, k2_path='k2_bin_58', num_original_maps_total= num_maps, heavy_aug = False, **args)
    Radio_val2 = RefractLunarLoader2(phase="val", root_dir=data_dir2, rm_path='rm58',freq=1, k2=False, k2_path='k2_bin_58', num_original_maps_total= num_maps, heavy_aug = False, **args)
    Radio_val3 = RefractLunarLoader(phase="val", root_dir=data_dir1, rm_path='rm415', freq = 0, k2=False, k2_path='k2_bin_415', num_original_maps_total= num_maps, heavy_aug = False, **args)
    Radio_val4 = RefractLunarLoader2(phase="val", root_dir=data_dir2, rm_path='rm415',freq=0, k2=False, k2_path='k2_bin_415', num_original_maps_total= num_maps, heavy_aug = False, **args)
    Radio_train1 = RefractLunarLoader(phase="train", root_dir=data_dir1, rm_path='rm58', freq = 1, k2=False, k2_path='k2_bin_58', num_original_maps_total= num_maps, heavy_aug = True, **args)
    Radio_train2 = RefractLunarLoader2(phase="train", root_dir=data_dir2, rm_path='rm58',freq=1, k2=False, k2_path='k2_bin_58', num_original_maps_total= num_maps, heavy_aug = True, **args)
    Radio_train3 = RefractLunarLoader(phase="train", root_dir=data_dir1, rm_path='rm415', freq = 0, k2=False, k2_path='k2_bin_415', num_original_maps_total= num_maps, heavy_aug = True, **args)
    Radio_train4 = RefractLunarLoader2(phase="train", root_dir=data_dir2, rm_path='rm415',freq=0, k2=False, k2_path='k2_bin_415', num_original_maps_total= num_maps, heavy_aug = True, **args)
    train_dataset = RadiomapWrapper(ConcatDataset([Radio_train1, Radio_train2, Radio_train3, Radio_train4]))
    val_dataset1 = RadiomapWrapper(ConcatDataset([Radio_val1, Radio_val2]))
    val_dataset2 = RadiomapWrapper(ConcatDataset([Radio_val3, Radio_val4]))
    return train_dataset, val_dataset1, val_dataset2


################################################################################
# VISUALIZATION 
################################################################################
def process_and_save_grid(conditioning, ground_truth, prediction, k2, path):
    """Normalizes and saves a grid of images for comparison."""
    prediction = torch.clip(prediction, 0, 1)
    conditioning = conditioning.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    prediction = prediction.cpu().numpy()
    k2 = k2.cpu().numpy()

    def normalize(img_array):
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = 255 * (img_array - img_min) / (img_max - img_min)
        else:
            img_array = np.zeros_like(img_array)
        return img_array.astype(np.uint8)

    cond_h, cond_fh, cond_tx = [normalize(conditioning[i]) for i in range(3)]
   

    gt_norm = normalize(ground_truth[0])
    pred_norm = normalize(prediction[0])
    k2_norm = normalize(k2[0])
    # print(k2_norm.shape)

    cond_h_rgb = np.stack([cond_h]*3, axis=-1)
    cond_fh_rgb = np.stack([cond_fh]*3, axis=-1)
    cond_tx_rgb = np.stack([cond_tx]*3, axis=-1)
    gt_rgb = np.stack([gt_norm]*3, axis=-1)
    pred_rgb = np.stack([pred_norm]*3, axis=-1)
    k2_rgb = np.stack([k2_norm]*3, axis=-1)
    # print(k2_rgb.shape)
    # Row 1: Heightmap, Filtered Heightmap, TX Location
    # Row 2: Ground Truth Radiomap, Predicted Radiomap, Empty Space
    grid_top = np.concatenate([cond_h_rgb, cond_fh_rgb, cond_tx_rgb], axis=1)
    grid_bottom = np.concatenate([gt_rgb, pred_rgb, k2_rgb], axis=1)
    grid = np.concatenate([grid_top, grid_bottom], axis=0)
    Image.fromarray(grid).save(path)



def train(model, k2_unet, train_loader, val_loader1, val_loader2, optimizer, scheduler, writer, cfg=None):
    best_loss = 1e10
    best_val = cfg.val_init 
    best_val58 = cfg.val_init
    best_val415 = cfg.val_init
    count = 0
    
    val_batch58 = next(iter(val_loader1))
    val_batch415 = next(iter(val_loader2))
    val_conditioning58 = val_batch58[0].cuda()
    val_ground_truth58 = val_batch58[1].cuda()
    val_k2_inputs58 = val_batch58[3].cuda()
    val_conditioning415 = val_batch415[0].cuda()
    val_ground_truth415 = val_batch415[1].cuda()
    val_k2_inputs415 = val_batch415[3].cuda()
    k2_unet.eval()
    with torch.no_grad():
        val_k2_map58 = torch.sigmoid(k2_unet(val_k2_inputs58))  
        val_k2_map415 = torch.sigmoid(k2_unet(val_k2_inputs415))# Get k2 prediction
    # looping over given number of epochs
    
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        tic = time.time()
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        model.train()
        train_losses = []
        for inputs, targets, _, k2_inputs in train_loader:
            count += 1

            inputs = inputs.cuda()
            k2_inputs = k2_inputs.cuda()
            targets = targets.cuda()
            k2_unet.eval()
            with torch.no_grad():
                k2_map = torch.sigmoid(k2_unet(k2_inputs))
            optimizer.zero_grad()
            model_input = torch.cat([inputs, k2_map], dim=1)
            preds = model(model_input)
            loss = MSE(preds, targets)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_postfix(loss_avg=f"{np.mean(train_losses):.6f}")
            # tensorboard logging
            writer.add_scalar('Train/Loss', loss.item(), count)

            if count % 1000 == 0:
                print(f'Epoch:{epoch}, Step:{count}, Loss:{loss.item():.6f}, BestVal:{best_val:.6f}, Time:{time.time()-tic}')
            tic = time.time()

        print(f"lr: {optimizer.param_groups[0]['lr']} at epoch {epoch}")
        scheduler.step()
        if epoch%cfg.val_freq==0:
            val_loss58, best_val58 = eval_model(model, k2_unet, val_loader1, error='MSE', best_val=best_val58, cfg=cfg)
            val_loss415, best_val415 = eval_model(model, k2_unet, val_loader2, error='MSE', best_val=best_val415, cfg=cfg)
            avg_loss = (val_loss58 + val_loss415)/2
            print("Train Loss: ", np.mean(train_losses))
            print("Val Loss: ", avg_loss)
            print("5.8GHz Val Loss: ", val_loss58)
            print("415MHz Loss: ", val_loss415)
            if avg_loss < best_val:
                best_val = avg_loss
                best_val415 = val_loss415
                best_val58 = val_loss58

                torch.save(model.state_dict(), f'{RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
                print(f'[*] model saved to: {RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
                f_log.write(f'[*] model saved to: {RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
                f_log.write('\n')
            model.eval()
            val_inputs58 = torch.cat([val_conditioning58, val_k2_map58], dim=1)
            val_inputs415 = torch.cat([val_conditioning415, val_k2_map415], dim=1)
            with torch.no_grad():
                pred_image58 = model(val_inputs58)
                pred_image415 = model(val_inputs415)
            img_path58 = os.path.join(RESULT_FOLDER, f"epoch_{epoch+1:04d}_sample58.png")
            img_path415 = os.path.join(RESULT_FOLDER, f"epoch_{epoch+1:04d}_sample415.png")
            
            wandb_image58 = process_and_save_grid(
                val_conditioning58[0], val_ground_truth58[0], pred_image58[0], val_k2_map58[0], img_path58
            )
            wandb_image415 = process_and_save_grid(
                val_conditioning415[0], val_ground_truth415[0], pred_image415[0], val_k2_map415[0], img_path415
            )
            print(f"Saved sample image to {img_path58}")
            print(f"Saved sample image to {img_path415}")

            writer.add_scalar('Val/Loss', avg_loss, count)

    return best_val

def eval_model(model, k2_unet, test_loader, error="MSE", best_val=100, cfg=None):

    # Set model to evaluate mode
    model.eval()

    n_samples = 0
    avg_loss = 0

    # check dataset type
    pred_cnt=1 # start from 1
    for inputs, targets, _, k2_inputs in tqdm(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        k2_inputs= k2_inputs.cuda()
        k2_unet.eval()
        with torch.no_grad():
            k2_map = torch.sigmoid(k2_unet(k2_inputs))
        model_input = torch.cat([inputs, k2_map], dim=1)
        with torch.set_grad_enabled(False):
            if error == "MSE":
                criterion = MSE
            elif error == "RMSE":
                criterion = RMSE
            elif error == "L1_loss":
                criterion = L1_loss

            preds = model(model_input)
            preds = torch.clip(preds, 0, 1)

            loss = criterion(preds, targets)
            # NMSE

            avg_loss += (loss.item() * inputs.shape[0])
            n_samples += inputs.shape[0]

    avg_loss = avg_loss / (n_samples + 1e-7)
    
    

    model.train()
    return avg_loss, best_val

class RadiomapWrapper(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        inputs, gain_tensor, name = self.original_dataset[idx]

        k2_inputs = inputs.clone()

        return (inputs, gain_tensor, name, k2_inputs)
    


def load_config_module(module_name, class_name):
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        return config_class()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, help='Directory where data located.')
    parser.add_argument('-n', '--network', type=str, help='Type of pmnet. pmnet_v1, pmnet_v3')
    parser.add_argument('-c', '--config', type=str, help='Class name in config file.')
    args = parser.parse_args()

    print('start')
    cfg = load_config_module(f'config.{args.config}', args.config)
    print(cfg.get_train_parameters())

    cfg.now = datetime.today().strftime("%Y%m%d%H%M") # YYYYmmddHHMM


    cfg.param_str = f'{cfg.batch_size}_{cfg.lr}_{cfg.lr_decay}_{cfg.step}'
    os.makedirs(TENSORBOARD_PREFIX, exist_ok=True)
    os.makedirs(f'{RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}', exist_ok=True)

    print('cfg.exp_name: ', cfg.exp_name)
    print('cfg.now: ', cfg.now)
    for k, v in cfg.get_train_parameters().items():
      print(f'{k}: {v}')
    print('RESULT_FOLDER: ', RESULT_FOLDER)
    print('cfg.param_str: ', cfg.param_str)

    # write config on the log file
    f_log = open(f'{RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/train.log', 'w')
    f_log.write(f'Train started at {cfg.now}.\n')
    for k, v in cfg.get_train_parameters().items():
      f_log.write(f'{k}: {v}\n')


    writer = SummaryWriter(log_dir=f'{TENSORBOARD_PREFIX}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}')
    if 'lunar' in args.config.lower():
        data_dir2 = '/mnt/2ndSSD/refraction_dataset_fin2'
        data_dir1=  '/mnt/2ndSSD/refraction_dataset_fin1'
        train_dataset, val_dataset1, val_dataset2 = load_lunar_for_hz(data_dir1, data_dir2)
        train_loader =  DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
        val_loader1 =  DataLoader(val_dataset1, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
        val_loader2 = DataLoader(val_dataset2, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
    # Load dataset
    
    # Initialize PMNet and Load pre-trained weights if given.
    if 'pmnet_v3' == args.network:
        from pmnet.models.pmnet_v3 import PMNet as Model
        # init model 
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
            in_ch=5)

        model.cuda()
    
    # Load pre-trained weights if given
    if hasattr(cfg, 'pre_trained_model'):
        print("Loading pretrained model from: ", cfg.pre_trained_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(cfg.pre_trained_model))
        model.to(device)


    k2_unet = K2_UNet(inputs=4, k2_bin=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k2_unet.load_state_dict(torch.load('/mnt/2ndSSD/k2_unet_hz/best_k2_model.pth', map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)
    print("Starting from epoch: ", cfg.start_epoch+1)
    for i in range(cfg.start_epoch):
        scheduler.step()

    best_val = train(model, k2_unet, train_loader, val_loader1, val_loader2, optimizer, scheduler, writer, cfg=cfg)
   
    print('[*] train ends... ')
    print(f'[*] best val loss: {best_val}')

    f_log.write(f'Train finished at {datetime.today().strftime("%Y%m%d%H%M")}.\n')
    f_log.close()
