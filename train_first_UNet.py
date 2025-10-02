
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import numpy as np
from PIL import Image


from k2net.diff_modules import K2_UNet
from dataset.refraction_loaders_mhz import RefractLunarLoader, RefractLunarLoader2


################################################################################
# 1. CONFIGURATION
################################################################################
class TrainingConfig:
    image_size = 256
    in_channels = 4 
    out_channels = 1 #
    k2_binary_target = True 


    train_batch_size = 16
    val_batch_size = 16
    num_epochs = 200
    learning_rate = 5e-5

    data_dir2 = '/mnt/2ndSSD/refraction_dataset_fin2'
    data_dir1 = '/mnt/2ndSSD/refraction_dataset_fin1'
    
    output_dir = "results/train_k2net"

config = TrainingConfig()




################################################################################
# 2. DATA LOADING
################################################################################

def load_lunar_for_k2_hz(data_dir1, data_dir2):
    args = {
        'train_split_ratio': 0.7, 'val_split_ratio': 0.15,
        'use_los_input': False, 'use_pathloss_input': False,
        'heavy_aug': True, 'non_global_norm': False,
        'filt_hm': False, 'use_filt_and_norm': True,
        'los_weight': .2, 'crop_resize': False, 'tx_array': False
    }
    num_maps = 500


    Radio_train1 = RefractLunarLoader(phase="train", root_dir=data_dir1, rm_path='rm58', freq=1, k2=True, k2_path='k2_bin_58', num_original_maps_total= num_maps, **args)
    Radio_val1 = RefractLunarLoader(phase="val", root_dir=data_dir1, rm_path='rm58', freq = 1, k2=True, k2_path='k2_bin_58', num_original_maps_total= num_maps, **args)

    Radio_train2 = RefractLunarLoader2(phase="train", root_dir=data_dir2, rm_path='rm58',freq=1, k2=True, k2_path='k2_bin_58', num_original_maps_total= num_maps, **args)
    Radio_val2 = RefractLunarLoader2(phase="val", root_dir=data_dir2, rm_path='rm58',freq=1, k2=True, k2_path='k2_bin_58', num_original_maps_total= num_maps, **args)
    
    Radio_train3 = RefractLunarLoader(phase="train", root_dir=data_dir1, rm_path='rm415', freq=0, k2=True, k2_path='k2_bin_415', num_original_maps_total= num_maps, **args)
    Radio_val3 = RefractLunarLoader(phase="val", root_dir=data_dir1, rm_path='rm415', freq = 0, k2=True, k2_path='k2_bin_415', num_original_maps_total= num_maps, **args)

    Radio_train4 = RefractLunarLoader2(phase="train", root_dir=data_dir2, rm_path='rm415',freq=0, k2=True, k2_path='k2_bin_415', num_original_maps_total= num_maps, **args)
    Radio_val4 = RefractLunarLoader2(phase="val", root_dir=data_dir2, rm_path='rm415',freq=0, k2=True, k2_path='k2_bin_415', num_original_maps_total= num_maps, **args)


    train_dataset = ConcatDataset([Radio_train1, Radio_train2, Radio_train3, Radio_train4])
    val_dataset1 = ConcatDataset([Radio_val1,Radio_val2])
    val_dataset2 = ConcatDataset([Radio_val3, Radio_val4])

    return train_dataset, val_dataset1, val_dataset2



################################################################################
# 3. TRAINING LOOP
################################################################################
def train():
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")
    
    model = K2_UNet(inputs=config.in_channels, k2_bin=config.k2_binary_target)
    model.load_state_dict(torch.load(f'{config.output_dir}/best_k2_model.pth'))

    train_dataset, val_dataset1, val_dataset2 = load_lunar_for_k2_hz(config.data_dir1, config.data_dir2)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=8)
    val_dataloader1 = DataLoader(val_dataset1, batch_size=config.val_batch_size, shuffle=False, num_workers=8)
    val_dataloader2 = DataLoader(val_dataset2, batch_size=config.val_batch_size, shuffle=False, num_workers=8)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


    model, optimizer, train_dataloader, val_dataloader1, val_dataloader2 = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader1, val_dataloader2
    )

    print("Starting K2 U-Net training...")
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for step, batch in enumerate(train_dataloader):
            inputs, targets, _ = batch
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        
        # --- Validation ---
        model.eval()
        val_loss58 = 0.0
        val_loss415=0.0
        with torch.no_grad():
            for batch in val_dataloader1:
                inputs, targets, _ = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss58 += loss.item()
            for batch in val_dataloader2:
                inputs, targets, _ = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss415 += loss.item()

        avg_val_loss58 = val_loss58 / len(val_dataloader1)

        avg_val_loss415 = val_loss415 / len(val_dataloader2)
        avg_val_loss = (avg_val_loss58+avg_val_loss415)/2
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"5.8GHz Loss: {avg_val_loss58:.6f}, 415MHz Loss: {avg_val_loss415:.6f}")


        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            model_path = os.path.join(config.output_dir, "best_k2_model.pth")
            accelerator.save(unwrapped_model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train()




