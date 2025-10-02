# THIS CODE WAS MODIFEID FROM PMNET
# https://github.com/abman23/pmnet


from __future__ import print_function, division
import os
import time

import warnings
warnings.filterwarnings("ignore")


import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled

from tqdm import tqdm
from datetime import datetime
import sys

import argparse
import importlib
import json
from pmnet.eval.eval import L1_loss, MSE, NMSE, RMSE, SSIM, PSNR
from k2net.diff_modules import K2_UNet
from dataset.refraction_loaders_mhz import RefractLunarLoader, RefractLunarLoader2

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


RESULT_FOLDER = './results/pmnet_results'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'

class RadiomapWrapper(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        inputs, gain_tensor, name = self.original_dataset[idx]

        k2_inputs = inputs.clone()

        return (inputs, gain_tensor, name, k2_inputs)
    
def save_results_grid(conditioning, ground_truth, prediction, k2_map, output_path):
    """Saves a 2x3 grid of images for visual comparison."""

    criterion = torch.nn.MSELoss()
    cond = (conditioning.cpu()).numpy()
    gt = (ground_truth.cpu()).numpy().squeeze()
    pred = (prediction.cpu()).numpy().squeeze()
    k2 = (k2_map.cpu()).numpy().squeeze()
    viz_mse = criterion(torch.from_numpy(gt), torch.from_numpy(pred)).item()
    viz_nmse = criterion(torch.from_numpy(gt), torch.from_numpy(pred))/criterion(torch.from_numpy(gt),0*torch.from_numpy(gt))
    viz_ssim, _= ssim(gt, pred, data_range=1.0, full=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Model Inference Results, MSE: {np.round(viz_mse, 5)}, NMSE: {np.round(viz_nmse.item(), 5)}, SSIM: {np.round(viz_ssim, 4)}", fontsize=16)

    
    axes[0, 0].imshow(cond[0], cmap='terrain')
    axes[0, 0].set_title("Condition: Heightmap")
    axes[0, 1].imshow(cond[3], cmap='terrain') 
    axes[0, 1].set_title("Condition: Filtered Heightmap")
    axes[0, 2].imshow(cond[1], cmap='viridis') 
    axes[0, 2].set_title("Condition: TX Location")

    axes[1, 0].imshow(gt, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title("Ground Truth Radiomap")
    axes[1, 1].imshow(pred, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title("Predicted Radiomap")
    axes[1, 2].imshow(k2, cmap='viridis')
    axes[1, 2].set_title("Predicted k^2 Map")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")

def load_lunar_for_hz(data_dir1, data_dir2):
    args = {
        'train_split_ratio': 0.7, 'val_split_ratio': 0.15,
        'use_los_input': False, 'use_pathloss_input': False,
        'non_global_norm': False,
        'filt_hm': False, 'use_filt_and_norm': True,
        'los_weight': .2, 'crop_resize': False, 'tx_array': False
    }
    num_maps = 500


    Radio_val1 = RefractLunarLoader(phase="test", root_dir=data_dir1, 
                                    rm_path='rm58', freq = 1, 
                                    k2=False, k2_path='k2_bin_58', 
                                    num_original_maps_total= num_maps, 
                                    heavy_aug = False, **args)
    Radio_val2 = RefractLunarLoader2(phase="test", root_dir=data_dir2, 
                                     rm_path='rm58',freq=1, 
                                     k2=False, k2_path='k2_bin_58', 
                                     num_original_maps_total= num_maps, 
                                     heavy_aug = False, **args)
    Radio_val3 = RefractLunarLoader(phase="test", root_dir=data_dir1, 
                                    rm_path='rm415', freq = 0, 
                                    k2=False, k2_path='k2_bin_415', 
                                    num_original_maps_total= num_maps, 
                                    heavy_aug = False, **args)
    Radio_val4 = RefractLunarLoader2(phase="test", root_dir=data_dir2, 
                                     rm_path='rm415',freq=0, 
                                     k2=False, k2_path='k2_bin_415', 
                                     num_original_maps_total= num_maps, 
                                     heavy_aug = False, **args)
    
    Radio_train1 = RefractLunarLoader(phase="train", root_dir=data_dir1, 
                                      rm_path='rm58', freq = 1, 
                                      k2=False, k2_path='k2_bin_58', 
                                      num_original_maps_total= num_maps, 
                                      heavy_aug = True, **args)
    Radio_train2 = RefractLunarLoader2(phase="train", root_dir=data_dir2, 
                                       rm_path='rm58',freq=1, 
                                       k2=False, k2_path='k2_bin_58', 
                                       num_original_maps_total= num_maps, 
                                       heavy_aug = True, **args)
    Radio_train3 = RefractLunarLoader(phase="train", root_dir=data_dir1, 
                                      rm_path='rm415', freq = 0, 
                                      k2=False, k2_path='k2_bin_415', 
                                      num_original_maps_total= num_maps, 
                                      heavy_aug = True, **args)
    Radio_train4 = RefractLunarLoader2(phase="train", root_dir=data_dir2, 
                                       rm_path='rm415',freq=0, 
                                       k2=False, k2_path='k2_bin_415', 
                                       num_original_maps_total= num_maps, 
                                       heavy_aug = True, **args)


    train_dataset = RadiomapWrapper(ConcatDataset([Radio_train1, Radio_train2, Radio_train3, Radio_train4]))
    val_dataset1 = RadiomapWrapper(ConcatDataset([Radio_val1, Radio_val2]))
    val_dataset2 = RadiomapWrapper(ConcatDataset([Radio_val3, Radio_val4]))

    return train_dataset, val_dataset1, val_dataset2

def eval_model(model, k2_unet, test_loader, error="MSE", cfg=None, infer_img_path=''):

 
    model.eval()

    n_samples = 0
    avg_mse = 0
    avg_ssim = 0
    avg_nmse = 0
    avg_psnr = 0
    k2_unet.eval()

    pred_cnt=0 
    for inputs, targets, _, k2_inputs in tqdm(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        k2_inputs= k2_inputs.cuda()
        

        with torch.no_grad():
            k2_map = torch.sigmoid(k2_unet(k2_inputs))

        model_input = torch.cat([inputs, k2_map], dim=1)
        with torch.set_grad_enabled(False):


            preds = model(model_input)
            preds = torch.clip(preds, 0, 1)

            mse_l = MSE(preds, targets)
            
            nmse_l = NMSE(preds, targets)
            
            ssim_l = SSIM(preds, targets)# NMSE
            psnr_l = PSNR(preds, targets)

            avg_mse += (mse_l.item() * inputs.shape[0])
            avg_nmse += (nmse_l.item() * inputs.shape[0])
            avg_ssim += (ssim_l.item() * inputs.shape[0])
            avg_psnr += (psnr_l.item() * inputs.shape[0])

            n_samples += inputs.shape[0]
        
        if pred_cnt % 50 == 0:
                output_path = os.path.join(RESULT_FOLDER, f"test_sample_visualization_{pred_cnt}_415.png")

                save_results_grid(
                    inputs[0], targets[0], preds[0], k2_map[0], output_path
                )
        pred_cnt+=1

    #     # inference image
    #     if infer_img_path!='':
    #         for i in range(len(preds)):
    #             plt.imshow(preds[i][0].cpu().detach().numpy(), cmap='viridis')

    #             img_name=os.path.join(infer_img_path,'inference_images',f'pred_{pred_cnt}.png')
    #             plt.savefig(img_name)
                
    #             plt.imshow(targets[i][0].cpu().detach().numpy(), cmap= 'viridis')

    #             img_name=os.path.join(infer_img_path,'inference_images',f'target_{pred_cnt}.png')
    #             plt.savefig(img_name)
    #             pred_cnt+=1
    #             if pred_cnt%100==0:
    #                 print(f'{img_name} saved')


    #         loss = criterion(preds, targets)
    #         # NMSE

    #         avg_loss += (loss.item() * inputs.shape[0])
    #         n_samples += inputs.shape[0]

    # avg_loss = avg_loss / (n_samples + 1e-7)
    avg_final_mse = avg_mse / (n_samples + 1e-7)
    avg_final_nmse = avg_nmse / (n_samples + 1e-7)
    avg_final_ssim = avg_ssim / (n_samples + 1e-7)
    avg_final_psnr = avg_psnr / (n_samples + 1e-7)
    return (avg_final_mse, avg_final_nmse, avg_final_ssim, avg_final_psnr)

def load_config_module(module_name, class_name):
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        return config_class()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default = '.', help='Directory where data located.')
    parser.add_argument('-n', '--network', type=str, default='pmnet_v3', help='Type of pmnet. pmnet_v1, pmnet_v3')
    parser.add_argument('-m', '--model_to_eval', default= './pretrained_models/pmnet/best_pm_model.pt',type=str, help='Pretrained model to evaluate.')
    parser.add_argument('-c', '--config', type=str, default='config_lunar_pmnet_V3', help='Class name in config file.')

    args = parser.parse_args()
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print('start')
    cfg = load_config_module(f'pmnet.config.{args.config}', args.config)
    print(cfg.get_train_parameters())
    cfg.now = datetime.today().strftime("%Y%m%d%H%M") 

    # Load dataset
    if 'lunar' in args.config.lower():
        data_dir1 = '/mnt/2ndSSD/refraction_dataset_fin1'
        data_dir2 = '/mnt/2ndSSD/refraction_dataset_fin2'
        train_dataset, test_dataset1, test_dataset2 = load_lunar_for_hz(data_dir1, data_dir2)
        train_loader =  DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
        test_loader1 =  DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=8, generator=torch.Generator(device='cuda'))
        test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=8, generator=torch.Generator(device='cuda'))
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
    
    # Load pre-trained weights to evaluate
    k2_unet = K2_UNet(inputs=4, k2_bin=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k2_unet.load_state_dict(torch.load('./pretrained_models/k2unet/best_k2_model.pth', map_location=device))
    model.load_state_dict(torch.load(args.model_to_eval))
    model.to(device)

     # create inference images directory if not exist
    os.makedirs(os.path.join(os.path.split(args.model_to_eval)[-2], 'inference_images'), exist_ok=True)

    result = eval_model(model, k2_unet, test_loader2, error="SSIM", cfg=None, 
                        infer_img_path=os.path.split(args.model_to_eval)[-2])
    result_json_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'result.json')
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print('Evaluation score(SSIM): ', result)