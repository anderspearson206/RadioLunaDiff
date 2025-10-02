import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import numpy as np
from PIL import Image

from k2net.diff_modules import K2_UNet
from dataset.refraction_loaders_mhz import RefractLunarLoader, RefractLunarLoader2

class TestConfig:
    model_path = "./pretrained_models/k2unet/best_k2_model.pth"

    data_dir2 = '/mnt/2ndSSD/refraction_dataset_fin2'
    data_dir1 = '/mnt/2ndSSD/refraction_dataset_fin1'

    # Output
    output_dir = "results/k2_test_outputs"
    num_test_samples = 10

config = TestConfig()


def save_comparison_image(inputs, ground_truth, prediction, path):
    """Saves a grid of images for visual comparison."""

    inputs_np = inputs.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    pred_np = prediction.cpu().numpy()
    print(inputs_np[2][0][0])
    def to_img(data):
        """Normalize and convert to uint8 image format."""
        data = (data - data.min()) / (data.max() - data.min()) 
        return (data * 255).astype(np.uint8)

    heightmap = to_img(inputs_np[0])
    filtered_heightmap = to_img(inputs_np[1])
    tx_location = (inputs_np[2] * 255).astype(np.uint8)

    h_rgb = np.stack([heightmap]*3, axis=-1)
    fh_rgb = np.stack([filtered_heightmap]*3, axis=-1)
    tx_rgb = np.stack([tx_location]*3, axis=-1)
    gt_rgb = np.stack([to_img(gt_np[0])]*3, axis=-1)
    pred_rgb = np.stack([to_img(pred_np[0])]*3, axis=-1)

    comparison_grid = np.concatenate([h_rgb, fh_rgb, tx_rgb, gt_rgb, pred_rgb], axis=1)
    
    # Save the image
    path = path[:-4] + '__'+str(int(np.average(tx_rgb))) + path[-4:]


    Image.fromarray(comparison_grid).save(path)


def load_lunar_for_k2_hz(data_dir1, data_dir2):
    args = {
        'train_split_ratio': 0.7, 'val_split_ratio': 0.15,
        'use_los_input': False, 'use_pathloss_input': False,
        'heavy_aug': False, 'non_global_norm': False,
        'filt_hm': False, 'use_filt_and_norm': True,
        'los_weight': .2, 'crop_resize': False, 'tx_array': False
    }
    num_maps = 500

    Radio_val1 = RefractLunarLoader(phase="test", root_dir=data_dir1, rm_path='rm58', freq = 1, k2=True, k2_path='k2_bin_58', num_original_maps_total= num_maps, **args)

    Radio_val2 = RefractLunarLoader2(phase="test", root_dir=data_dir2, rm_path='rm58',freq=1, k2=True, k2_path='k2_bin_58', num_original_maps_total= num_maps, **args)

    Radio_val3 = RefractLunarLoader(phase="test", root_dir=data_dir1, rm_path='rm415', freq = 0, k2=True, k2_path='k2_bin_415', num_original_maps_total= num_maps, **args)

    Radio_val4 = RefractLunarLoader2(phase="test", root_dir=data_dir2, rm_path='rm415',freq=0, k2=True, k2_path='k2_bin_415', num_original_maps_total= num_maps, **args)

    val_dataset = ConcatDataset([Radio_val1, Radio_val2, Radio_val3, Radio_val4])

    return val_dataset



def test():
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {config.model_path}")
    model = K2_UNet(inputs=4, k2_bin=True)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    

    test_dataset = load_lunar_for_k2_hz(config.data_dir1, config.data_dir2)
    dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


    print(f"Generating {config.num_test_samples} test images...")
    with torch.no_grad():
        for i, (inputs, targets, names) in enumerate(dataloader):
            if i >= config.num_test_samples:
                break
            
            inputs = inputs.to(device)
            
            # Get model prediction
            logits = model(inputs)
            # Apply sigmoid to convert logits to probabilities
            predictions = torch.sigmoid(logits) 
            # binary_predictions = (predictions>=0.5).float()
            binary_predictions = predictions
            # Get the first item from the batch
            single_input = inputs[0]
            single_target = targets[0]
            single_prediction = binary_predictions[0]
            
            # Save the comparison image
            output_path = os.path.join(config.output_dir, f"sample_{i:03d}_{names[0]}.png")
            save_comparison_image(single_input, single_target, single_prediction, output_path)

    print(f"Done! Test images saved to '{config.output_dir}'.")
    print("Each image shows: Heightmap, Filtered Heightmap, TX Location, Ground Truth K², Predicted K²")

if __name__ == "__main__":
    test()
