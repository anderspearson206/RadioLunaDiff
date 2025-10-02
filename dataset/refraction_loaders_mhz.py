from __future__ import print_function, division
import os
import torch
#import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils, datasets, models
import warnings
#import math
#from PIL import Image
from torchvision.transforms import functional as TF # Added for functional transforms
import random # Added for random decisions
HM_GLOBAL_MIN = 0
HM_GLOBAL_MAX = 496
RM_GLOBAL_MIN = -200
RM_GLOBAL_MAX = 10
FHM_GLOBAL_MAX = 140
FHM_GLOBAL_MIN = 0

warnings.filterwarnings("ignore")

class RefractLunarLoader(Dataset):
    """
    Dataset loader for augmented heightmap, radio map (gain), TX location,
    and optionally Line of Sight (LOS) map. Data stored as .npy files.
    It determines which files to load based on an overall index, the number of
    original maps, and the number of augmentations per map. It also handles
    splitting data into train/validation/test phases based on original map indices.
    """

    def __init__(self, root_dir,
                 phase="train",
                 rm_path ='rm',
                 freq = 0,
                 k2 = False,
                 k2_path = 'k2', 
                 num_original_maps_total=302,
                 split_maps=16,
                 train_split_ratio=0.7,
                 val_split_ratio=0.15, # Test split will be the remainder
                 random_seed=42,
                 hm_norm_params=(HM_GLOBAL_MIN, HM_GLOBAL_MAX),
                 rm_norm_params=(RM_GLOBAL_MIN, RM_GLOBAL_MAX),
                 fhm_norm_params=(FHM_GLOBAL_MIN, FHM_GLOBAL_MAX),
                 k2_norm_params=(0,1),
                 pregen_k2=False,
                 pregen_k2_path = 'k2_pregen',
                 use_los_input=False, 
                 use_pathloss_input = False,
                 transform=None, 
                 heavy_aug = False,
                 non_global_norm= False, 
                 verbose = False, 
                 los_weight = 1, 
                 filt_hm = False,
                 use_filt_and_norm=False, 
                 crop_resize = False, 
                 tx_array = False
                 ):
        """
        Args:
            root_dir (string): Base directory where 'heightmaps', 'radiomaps',
                               'tx_locations', and optionally 'los_maps' subfolders are located.
            phase (string): "train", "val", or "test" to select the data split.
            num_original_maps_total (int): Total number of unique original maps
                                           before augmentation (e.g., files indexed 0 to 49).
            augmentations_per_map (int): Number of augmented samples created
                                           for each original map.
            train_split_ratio (float): Proportion of original maps to use for training.
            val_split_ratio (float): Proportion of original maps to use for validation.
            random_seed (int): Seed for shuffling original map indices for reproducible splits.
            hm_norm_params (tuple): (min_val, max_val) for heightmap normalization.
            rm_norm_params (tuple): (min_val, max_val) for radio map normalization.
            use_los_input (bool): If True, loads and includes the LOS map as an input channel.
                                   Defaults to False.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.freq = freq
        self.phase = phase 
        self.split_maps = split_maps
        self.transform = transform
        self.use_los_input = use_los_input
        self.use_pathloss_input = use_pathloss_input
        self.heavy_aug = heavy_aug
        self.non_global_norm = non_global_norm
        self.los_weight = los_weight
        self.verbose = verbose
        self.filt_hm = filt_hm
        self.use_filt_and_norm = use_filt_and_norm
        self.crop_resize = crop_resize
        self.tx_array = tx_array
        self.k2 = k2
        self.pregen_k2 = pregen_k2
        if self.filt_hm and not self.use_filt_and_norm:
            self.heightmap_dir = os.path.join(root_dir, 'filt_hm')
        else:
            self.heightmap_dir = os.path.join(root_dir, 'hm')
            
        if self.use_filt_and_norm and not self.filt_hm:
            self.filt_hm_dir = os.path.join(root_dir, 'filt_hm')
        
        if self.k2:
            self.radiomap_dir = os.path.join(root_dir, k2_path)
        else:
            self.radiomap_dir = os.path.join(root_dir, rm_path)
        self.tx_location_dir = os.path.join(root_dir, 'tx')
        if self.use_los_input:
            self.los_dir = os.path.join(root_dir, 'los') 
        if self.use_pathloss_input:
            self.pl_dir = os.path.join(root_dir, 'pl') 
        # Normalization parameters
        if self.pregen_k2:
            self.pregen_k2_dir = os.path.join(root_dir, pregen_k2_path)
            self.pregen_k2_min, self.pregen_k2_max = k2_norm_params
            self.pregen_k2_norm_range = self.pregen_k2_max -self.pregen_k2_min

        self.hm_min, self.hm_max = hm_norm_params
        if self.filt_hm and not self.use_filt_and_norm:
            self.hm_min, self.hm_max = fhm_norm_params
        elif self.use_filt_and_norm:
            self.fhm_min, self.fhm_max = fhm_norm_params
        if self.k2:
            self.rm_min, self.rm_max = k2_norm_params
        else:
            self.rm_min, self.rm_max = rm_norm_params
        self.hm_norm_range = (self.hm_max - self.hm_min)
        if self.hm_norm_range == 0:
            print("Warning: Heightmap normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
            self.hm_norm_range = 1.0

        self.rm_norm_range = (self.rm_max - self.rm_min)
        if self.rm_norm_range == 0:
            print("Warning: Radio map normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
            self.rm_norm_range = 1.0
        if self.use_filt_and_norm and not self.filt_hm:
            self.fhm_norm_range = (self.fhm_max - self.fhm_min)
            if self.fhm_norm_range == 0:
                print("Warning: filtered map normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
                self.fhm_norm_range = 1.0
        # Determine the original map indices for this phase (train/val/test)
        all_original_map_ids = np.arange(num_original_maps_total)

        np.random.seed(random_seed)
        np.random.shuffle(all_original_map_ids)

        n_train = int(np.floor(train_split_ratio * num_original_maps_total))
        n_val = int(np.floor(val_split_ratio * num_original_maps_total))

        if n_train + n_val > num_original_maps_total:
            print(f"Warning: Sum of train ({train_split_ratio*100}%) and val ({val_split_ratio*100}%) ratios exceeds 100%. Adjusting validation count.")
            n_val = num_original_maps_total - n_train
            if n_val < 0 : n_val = 0

        if phase == "train":
            self.current_phase_original_map_ids = all_original_map_ids[:n_train]
        elif phase == "val":
            self.current_phase_original_map_ids = all_original_map_ids[n_train : n_train + n_val]
        elif phase == "test":
            self.current_phase_original_map_ids = all_original_map_ids[n_train + n_val:]
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train', 'val', or 'test'.")

        if len(self.current_phase_original_map_ids) == 0 and num_original_maps_total > 0:
              print(f"Warning: Phase '{self.phase}' resulted in an empty set of original map IDs. "
                    f"Total original maps: {num_original_maps_total}, "
                    f"Train count: {n_train}, Val count: {n_val}. "
                    f"Consider adjusting split ratios or total map count.")

        print(f"Dataset phase: '{self.phase}', using {len(self.current_phase_original_map_ids)} original map IDs for this instance. LOS input: {self.use_los_input}")

    def __len__(self):
        # Total number of samples is the number of selected original maps for this phase * augmentations per map
        return len(self.current_phase_original_map_ids) * self.split_maps

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not (0 <= idx < self.__len__()):
            if self.__len__() == 0:
                raise IndexError(f"Cannot get item for index {idx} from an empty dataset (phase: {self.phase}). Check dataset initialization and splits.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__len__()}")

        original_map_group_idx = int(np.floor(idx / self.split_maps))
        augmentation_idx = idx % self.split_maps
        actual_original_map_id = self.current_phase_original_map_ids[original_map_group_idx]

        base_name = f"{actual_original_map_id}_{augmentation_idx}"
        if self.filt_hm and not self.use_filt_and_norm:
            hm_path = os.path.join(self.heightmap_dir, f'fhm_{base_name}.npy')
        else:
            hm_path = os.path.join(self.heightmap_dir, f'hm_{base_name}.npy')
        if self.use_filt_and_norm and not self.filt_hm:
            fhm_path = os.path.join(self.filt_hm_dir, f'fhm_{base_name}.npy')

        if self.k2:
            rm_path = os.path.join(self.radiomap_dir, f'k2_{base_name}.npy')
        else:
            rm_path = os.path.join(self.radiomap_dir, f'rm_{base_name}.npy')
        tx_path = os.path.join(self.tx_location_dir, f'tx_{base_name}.npy')
        los_path = None
        if self.use_los_input:
            los_path = os.path.join(self.los_dir, f'los_{base_name}.npy')
        if self.use_pathloss_input:
            pl_path = os.path.join(self.pl_dir, f'pl_{base_name}.npy')
        if self.pregen_k2:
            pk2_path = os.path.join(self.pregen_k2_dir, f'k2_{base_name}.npy')
        try:
            heightmap = np.load(hm_path)
 
            heightmap = heightmap - np.min(heightmap)
            if self.non_global_norm:
                self.hm_norm_range = np.max(heightmap)
            radiomap = np.load(rm_path)
            radiomap = np.nan_to_num(radiomap, nan=RM_GLOBAL_MIN) # Use the defined RM_GLOBAL_MIN
            tx_location = np.load(tx_path)
            if self.use_filt_and_norm and not self.filt_hm:
                filt_hm = np.load(fhm_path)
                filt_hm = filt_hm-np.min(filt_hm)
                if self.non_global_norm:
                    self.fhm_norm_range = np.max(filt_hm)
            if self.tx_array:
                tx_location = self.create_horizontal_line(tx_location, 3)

            los_map = None
            if self.use_los_input:
                los_map = np.load(los_path) * self.los_weight
            pl_map = None
            if self.use_pathloss_input:
                pl_map = np.load(pl_path) 
            if self.pregen_k2:
                pk2_map = np.load(pk2_path)
        except FileNotFoundError as e:
            missing_path = e.filename
            print(f"Error: File not found: {missing_path} (generated base_name '{base_name}').")
            print(f"  Index: {idx}, Original Map Group Idx: {original_map_group_idx}, Actual Original Map ID: {actual_original_map_id}, Augmentation Idx: {augmentation_idx}")
            if self.use_los_input and los_path and missing_path == los_path:
                 print(f"  Attempted to load LOS from: {los_path}")
            elif missing_path == hm_path:
                 print(f"  Attempted to load HM from: {hm_path}")
            elif missing_path == rm_path:
                 print(f"  Attempted to load RM from: {rm_path}")
            elif missing_path == tx_path:
                 print(f"  Attempted to load TX from: {tx_path}")
            raise
        except Exception as e:
            print(f"Error loading files for base_name {base_name}: {e}")
            raise
        except ValueError as e:
            print(f"ValError loading files for base_name {base_name}: {e}")
            raise


        # --- Normalization ---
        heightmap_normalized = np.clip((heightmap.astype(np.float32) - self.hm_min) / self.hm_norm_range, 0, 1)
        radiomap_normalized = np.clip((radiomap.astype(np.float32) - self.rm_min) / self.rm_norm_range, 0, 1)
        
        tx_location_processed = tx_location.astype(np.float32) # Assumed to be 0 or 1 already
        if self.use_filt_and_norm and not self.filt_hm:
            filt_hm_normalized = np.clip((filt_hm.astype(np.float32) - self.fhm_min) / self.fhm_norm_range, 0, 1)
        # --- Convert to PyTorch Tensors and add channel dimension ---
        heightmap_tensor = torch.from_numpy(heightmap_normalized).unsqueeze(0) # (1, H, W)
        tx_location_tensor = torch.from_numpy(tx_location_processed).unsqueeze(0) # (1, H, W)
        gain_tensor = torch.from_numpy(radiomap_normalized).unsqueeze(0) # (1, H, W)

        if self.use_filt_and_norm and not self.filt_hm:
            if filt_hm_normalized is None:
                raise ValueError(f"filtered heightmap is none for basename {base_name}")
            fhm_tensor = torch.from_numpy(filt_hm_normalized).unsqueeze(0)

        if self.use_los_input:
            if los_map is None: 
                raise ValueError(f"LOS map is None for base_name {base_name} even though use_los_input is True. This should not happen.")
            los_map_processed = los_map.astype(np.float32)
            los_tensor = torch.from_numpy(los_map_processed).unsqueeze(0) # (1, H, W)
            # input_channels.append(los_tensor)
        if self.use_pathloss_input:
            if pl_map is None: 
                raise ValueError(f"LOS map is None for base_name {base_name} even though use_los_input is True. This should not happen.")
            pl_map_proccessed = pl_map.astype(np.float32)
            pl_tensor = torch.from_numpy(pl_map_proccessed).unsqueeze(0) # (1, H, W)
   
        if self.pregen_k2:
            if pk2_map is None:
                raise ValueError(f'pregen k2 map for name {base_name} is None')
            pk2_map_processed = pk2_map.astype(np.float32)
            pk2_tensor = torch.from_numpy(pk2_map_processed).unsqueeze(0)

        if self.heavy_aug:
            if random.random() < 0.5:
                heightmap_tensor = TF.hflip(heightmap_tensor)
                tx_location_tensor = TF.hflip(tx_location_tensor)
                gain_tensor = TF.hflip(gain_tensor)
                if self.use_los_input:
                    los_tensor = TF.hflip(los_tensor)
                if self.use_pathloss_input:
                    pl_tensor = TF.hflip(pl_tensor)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.hflip(fhm_tensor)
                

            # Random Vertical Flip
            if random.random() < 0.5:
                heightmap_tensor = TF.vflip(heightmap_tensor)
                tx_location_tensor = TF.vflip(tx_location_tensor)
                gain_tensor = TF.vflip(gain_tensor)
                if self.use_los_input:
                    los_tensor = TF.vflip(los_tensor)
                if self.use_pathloss_input:
                    pl_tensor = TF.vflip(pl_tensor)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.vflip(fhm_tensor)
                    
            k = random.randint(0, 3)
            if k > 0:
                heightmap_tensor = TF.rotate(heightmap_tensor, angle=k*90)
                tx_location_tensor = TF.rotate(tx_location_tensor, angle=k*90)
                gain_tensor = TF.rotate(gain_tensor, angle=k*90)
                if self.use_los_input:
                    los_tensor = TF.rotate(los_tensor, angle=k*90)
                if self.use_pathloss_input:
                    pl_tensor = TF.rotate(pl_tensor, angle=k*90)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.rotate(fhm_tensor, angle = k*90)
                    
            if random.random() < 0.5:
                heightmap_tensor = torch.transpose(heightmap_tensor, 1, 2)
                tx_location_tensor = torch.transpose(tx_location_tensor, 1, 2)
                gain_tensor = torch.transpose(gain_tensor, 1, 2)
                if self.use_los_input:
                    los_tensor = torch.transpose(los_tensor, 1, 2)
                if self.use_pathloss_input:
                    pl_tensor = torch.transpose(pl_tensor, 1, 2)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = torch.transpose(fhm_tensor, 1, 2)
                    
            if self.crop_resize:   
                shape_aug = transforms.RandomResizedCrop(
                    size=(256, 256),
                    scale=(0.1, 1.0),
                    ratio=(1, 1)
                )
                # i, j, h, w = shape_aug.get_params(img=heightmap_tensor, scale=shape_aug.scale, ratio=shape_aug.ratio)
                # heightmap_tensor = TF.resized_crop(heightmap_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                # tx_location_tensor = TF.resized_crop(tx_location_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                # gain_tensor = TF.resized_crop(gain_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                # if self.use_los_input:
                #     los_tensor = TF.resized_crop(los_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)

                i, j, h, w = shape_aug.get_params(img=heightmap_tensor, scale=shape_aug.scale, ratio=shape_aug.ratio)

                tx_coords = tx_location_tensor.nonzero(as_tuple=False)

                if tx_coords.numel() > 0:

                    tx_y = tx_coords[0, 1].item()
                    tx_x = tx_coords[0, 2].item()
                    
                    img_height, img_width = heightmap_tensor.shape[-2:]

                    top_min = max(0, tx_y - h + 1)
                    top_max = min(img_height - h, tx_y)

                    left_min = max(0, tx_x - w + 1)
                    left_max = min(img_width - w, tx_x)

                    if top_max >= top_min:
                        i = torch.randint(top_min, top_max + 1, size=(1,)).item()
                        
                    if left_max >= left_min:
                        j = torch.randint(left_min, left_max + 1, size=(1,)).item()

                heightmap_tensor = TF.resized_crop(heightmap_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                tx_location_tensor = TF.resized_crop(tx_location_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                gain_tensor = TF.resized_crop(gain_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                if self.use_los_input:
                    los_tensor = TF.resized_crop(los_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                if self.use_pathloss_input:
                    pl_tensor = TF.resized_crop(pl_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.resized_crop(fhm_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                
        freq_tensor = torch.full_like(heightmap_tensor, float(self.freq))
        input_list = [heightmap_tensor, tx_location_tensor, freq_tensor]

        if self.use_filt_and_norm:
            input_list.append(fhm_tensor)
        if self.use_los_input:
            input_list.append(los_tensor)
        if self.use_pathloss_input:
            input_list.append(pl_tensor)

        inputs_tensor = torch.cat(input_list, dim=0)
            
            
        if self.transform:
            # Note: Standard torchvision transforms often expect PIL Image or (C,H,W) tensor.
            # If your custom transform expects something else, ensure compatibility.
            # The current setup provides inputs_tensor as (C, H, W) and gain_tensor as (1, H, W).
            inputs_tensor = self.transform(inputs_tensor) #.type(torch.float32) # type cast can be part of transform
            gain_tensor = self.transform(gain_tensor) #.type(torch.float32)

        # Ensure correct type after potential transform
        inputs_tensor = inputs_tensor.type(torch.float32)
        gain_tensor = gain_tensor.type(torch.float32)

        return (inputs_tensor, gain_tensor, base_name)
    
    def create_horizontal_line(self, array: np.ndarray, line_length: int) -> np.ndarray:
        new_array = array
        coords = np.where(new_array == 1)
        if coords[0].size == 0:
            print("Warning: No '1' found in the array.")
            return new_array
        row, col = coords[0][0], coords[1][0]
        half_length = (line_length - 1) // 2
        start_col = col - half_length
        end_col = col + half_length + 1
        array_width = new_array.shape[1]
        start_col_clipped = max(0, start_col)
        end_col_clipped = min(array_width, end_col)
        new_array[row, start_col_clipped:end_col_clipped] = 1
        return new_array
    
    
    


class RefractLunarLoader2(Dataset):
    """
    Dataset loader for augmented heightmap, radio map (gain), TX location,
    and optionally Line of Sight (LOS) map. Data stored as .npy files.
    It determines which files to load based on an overall index, the number of
    original maps, and the number of augmentations per map. It also handles
    splitting data into train/validation/test phases based on original map indices.
    """

    def __init__(self, root_dir,
                 phase="train",
                 rm_path = 'rm', 
                 freq = 0,
                 k2=False,
                 k2_path = 'k2',
                 num_original_maps_total=100,
                 split_maps=50,
                 train_split_ratio=0.7,
                 val_split_ratio=0.15, # Test split will be the remainder
                 random_seed=42,
                 hm_norm_params=(HM_GLOBAL_MIN, HM_GLOBAL_MAX),
                 rm_norm_params=(RM_GLOBAL_MIN, RM_GLOBAL_MAX),
                 fhm_norm_params=(FHM_GLOBAL_MIN, FHM_GLOBAL_MAX),
                 k2_norm_params=(0,1),
                 use_los_input=False, 
                 use_pathloss_input = False,
                 transform=None, 
                 heavy_aug = False,
                 non_global_norm= False, 
                 verbose = False, 
                 los_weight = 1, 
                 filt_hm = False,
                 use_filt_and_norm=False, 
                 crop_resize = False, 
                 tx_array = False
                 ):
        """
        Args:
            root_dir (string): Base directory where 'heightmaps', 'radiomaps',
                               'tx_locations', and optionally 'los_maps' subfolders are located.
            phase (string): "train", "val", or "test" to select the data split.
            num_original_maps_total (int): Total number of unique original maps
                                           before augmentation (e.g., files indexed 0 to 49).
            augmentations_per_map (int): Number of augmented samples created
                                           for each original map.
            train_split_ratio (float): Proportion of original maps to use for training.
            val_split_ratio (float): Proportion of original maps to use for validation.
            random_seed (int): Seed for shuffling original map indices for reproducible splits.
            hm_norm_params (tuple): (min_val, max_val) for heightmap normalization.
            rm_norm_params (tuple): (min_val, max_val) for radio map normalization.
            use_los_input (bool): If True, loads and includes the LOS map as an input channel.
                                   Defaults to False.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.freq = freq
        self.phase = phase 
        self.split_maps = split_maps
        self.transform = transform
        self.use_los_input = use_los_input
        self.use_pathloss_input = use_pathloss_input
        self.heavy_aug = heavy_aug
        self.non_global_norm = non_global_norm
        self.los_weight = los_weight
        self.verbose = verbose
        self.filt_hm = filt_hm
        self.use_filt_and_norm = use_filt_and_norm
        self.crop_resize = crop_resize
        self.tx_array = tx_array
        self.k2 = k2
        if self.filt_hm and not self.use_filt_and_norm:
            self.heightmap_dir = os.path.join(root_dir, 'filt_hm')
        else:
            self.heightmap_dir = os.path.join(root_dir, 'hm')
            
        if self.use_filt_and_norm and not self.filt_hm:
            self.filt_hm_dir = os.path.join(root_dir, 'filt_hm')
        
        if self.k2:
            self.radiomap_dir = os.path.join(root_dir, k2_path)
        else:
            self.radiomap_dir = os.path.join(root_dir, rm_path)
        self.tx_location_dir = os.path.join(root_dir, 'tx')
        if self.use_los_input:
            self.los_dir = os.path.join(root_dir, 'los') 
        if self.use_pathloss_input:
            self.pl_dir = os.path.join(root_dir, 'pl') 
        # Normalization parameters
        
        self.hm_min, self.hm_max = hm_norm_params
        if self.filt_hm and not self.use_filt_and_norm:
            self.hm_min, self.hm_max = fhm_norm_params
        elif self.use_filt_and_norm:
            self.fhm_min, self.fhm_max = fhm_norm_params
        if self.k2:
            self.rm_min, self.rm_max = k2_norm_params
        else:
            self.rm_min, self.rm_max = rm_norm_params
        self.hm_norm_range = (self.hm_max - self.hm_min)
        if self.hm_norm_range == 0:
            print("Warning: Heightmap normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
            self.hm_norm_range = 1.0

        self.rm_norm_range = (self.rm_max - self.rm_min)
        if self.rm_norm_range == 0:
            print("Warning: Radio map normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
            self.rm_norm_range = 1.0
        if self.use_filt_and_norm and not self.filt_hm:
            self.fhm_norm_range = (self.fhm_max - self.fhm_min)
            if self.fhm_norm_range == 0:
                print("Warning: filtered map normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
                self.fhm_norm_range = 1.0
        # Determine the original map indices for this phase (train/val/test)
        all_original_map_ids = np.arange(num_original_maps_total)

        np.random.seed(random_seed)
        np.random.shuffle(all_original_map_ids)

        n_train = int(np.floor(train_split_ratio * num_original_maps_total))
        n_val = int(np.floor(val_split_ratio * num_original_maps_total))

        if n_train + n_val > num_original_maps_total:
            print(f"Warning: Sum of train ({train_split_ratio*100}%) and val ({val_split_ratio*100}%) ratios exceeds 100%. Adjusting validation count.")
            n_val = num_original_maps_total - n_train
            if n_val < 0 : n_val = 0

        if phase == "train":
            self.current_phase_original_map_ids = all_original_map_ids[:n_train]
        elif phase == "val":
            self.current_phase_original_map_ids = all_original_map_ids[n_train : n_train + n_val]
        elif phase == "test":
            self.current_phase_original_map_ids = all_original_map_ids[n_train + n_val:]
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train', 'val', or 'test'.")

        if len(self.current_phase_original_map_ids) == 0 and num_original_maps_total > 0:
              print(f"Warning: Phase '{self.phase}' resulted in an empty set of original map IDs. "
                    f"Total original maps: {num_original_maps_total}, "
                    f"Train count: {n_train}, Val count: {n_val}. "
                    f"Consider adjusting split ratios or total map count.")

        print(f"Dataset phase: '{self.phase}', using {len(self.current_phase_original_map_ids)} original map IDs for this instance. LOS input: {self.use_los_input}")

    def __len__(self):
        # Total number of samples is the number of selected original maps for this phase * augmentations per map
        return len(self.current_phase_original_map_ids) * self.split_maps

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not (0 <= idx < self.__len__()):
            if self.__len__() == 0:
                raise IndexError(f"Cannot get item for index {idx} from an empty dataset (phase: {self.phase}). Check dataset initialization and splits.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__len__()}")

        original_map_group_idx = int(np.floor(idx / self.split_maps))
        augmentation_idx = idx % self.split_maps
        actual_original_map_id = self.current_phase_original_map_ids[original_map_group_idx]

        # base_name1 = f"{actual_original_map_id}_{augmentation_idx}"
        base_name = f"{actual_original_map_id}_{augmentation_idx}"
        if self.filt_hm and not self.use_filt_and_norm:
            hm_path = os.path.join(self.heightmap_dir, f'fhm_{actual_original_map_id}.npy')
        else:
            hm_path = os.path.join(self.heightmap_dir, f'hm_{actual_original_map_id}.npy')
        if self.use_filt_and_norm and not self.filt_hm:
            fhm_path = os.path.join(self.filt_hm_dir, f'fhm_{actual_original_map_id}.npy')
        if self.k2:
            rm_path = os.path.join(self.radiomap_dir, f'k2_{base_name}.npy')
        else:
            rm_path = os.path.join(self.radiomap_dir, f'rm_{base_name}.npy')
        tx_path = os.path.join(self.tx_location_dir, f'tx_{base_name}.npy')
        los_path = None
        if self.use_los_input:
            los_path = os.path.join(self.los_dir, f'los_{base_name}.npy')
        if self.use_pathloss_input:
            pl_path = os.path.join(self.pl_dir, f'pl_{base_name}.npy')

        try:
            heightmap = np.load(hm_path)
 
            heightmap = heightmap - np.min(heightmap)
            if self.non_global_norm:
                self.hm_norm_range = np.max(heightmap)
            try:

                radiomap = np.load(rm_path)
            except ValueError as e:
                print('!!!!!!! FAILED TO LOAD FILE (VALUE ERROR)!!!!')
                print(f'PATH: {rm_path}')
                print(f'Error: {e}')
                raise e
            radiomap = np.nan_to_num(radiomap, nan=RM_GLOBAL_MIN) # Use the defined RM_GLOBAL_MIN
            tx_location = np.load(tx_path)
            if self.use_filt_and_norm and not self.filt_hm:
                filt_hm = np.load(fhm_path)
                filt_hm = filt_hm-np.min(filt_hm)
                if self.non_global_norm:
                    self.fhm_norm_range = np.max(filt_hm)
            if self.tx_array:
                tx_location = self.create_horizontal_line(tx_location, 3)

            los_map = None
            if self.use_los_input:
                los_map = np.load(los_path) * self.los_weight
            pl_map = None
            if self.use_pathloss_input:
                pl_map = np.load(pl_path) 

        except FileNotFoundError as e:
            missing_path = e.filename
            print(f"Error: File not found: {missing_path} (generated base_name '{base_name}').")
            print(f"  Index: {idx}, Original Map Group Idx: {original_map_group_idx}, Actual Original Map ID: {actual_original_map_id}, Augmentation Idx: {augmentation_idx}")
            if self.use_los_input and los_path and missing_path == los_path:
                 print(f"  Attempted to load LOS from: {los_path}")
            elif missing_path == hm_path:
                 print(f"  Attempted to load HM from: {hm_path}")
            elif missing_path == rm_path:
                 print(f"  Attempted to load RM from: {rm_path}")
            elif missing_path == tx_path:
                 print(f"  Attempted to load TX from: {tx_path}")
            raise
        except Exception as e:
            print(f"Error loading files for base_name {base_name}: {e}")
            raise
        except:
            print(f"Error loading files for base_name {base_name}")
            raise

        # --- Normalization ---
        heightmap_normalized = (heightmap.astype(np.float32) - self.hm_min) / self.hm_norm_range
        radiomap_normalized = (radiomap.astype(np.float32) - self.rm_min) / self.rm_norm_range
        
        tx_location_processed = tx_location.astype(np.float32) # Assumed to be 0 or 1 already
        if self.use_filt_and_norm and not self.filt_hm:
            filt_hm_normalized = (filt_hm.astype(np.float32) - self.fhm_min) / self.fhm_norm_range
        # --- Convert to PyTorch Tensors and add channel dimension ---
        heightmap_tensor = torch.from_numpy(heightmap_normalized).unsqueeze(0) # (1, H, W)
        tx_location_tensor = torch.from_numpy(tx_location_processed).unsqueeze(0) # (1, H, W)
        gain_tensor = torch.from_numpy(radiomap_normalized).unsqueeze(0) # (1, H, W)

        if self.use_filt_and_norm and not self.filt_hm:
            if filt_hm_normalized is None:
                raise ValueError(f"filtered heightmap is none for basename {base_name}")
            fhm_tensor = torch.from_numpy(filt_hm_normalized).unsqueeze(0)

        if self.use_los_input:
            if los_map is None: 
                raise ValueError(f"LOS map is None for base_name {base_name} even though use_los_input is True. This should not happen.")
            los_map_processed = los_map.astype(np.float32)
            los_tensor = torch.from_numpy(los_map_processed).unsqueeze(0) # (1, H, W)
            # input_channels.append(los_tensor)
        if self.use_pathloss_input:
            if pl_map is None: 
                raise ValueError(f"LOS map is None for base_name {base_name} even though use_los_input is True. This should not happen.")
            pl_map_proccessed = pl_map.astype(np.float32)
            pl_tensor = torch.from_numpy(pl_map_proccessed).unsqueeze(0) # (1, H, W)
   
        
        if self.heavy_aug:
            if random.random() < 0.5:
                heightmap_tensor = TF.hflip(heightmap_tensor)
                tx_location_tensor = TF.hflip(tx_location_tensor)
                gain_tensor = TF.hflip(gain_tensor)
                if self.use_los_input:
                    los_tensor = TF.hflip(los_tensor)
                if self.use_pathloss_input:
                    pl_tensor = TF.hflip(pl_tensor)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.hflip(fhm_tensor)
                

            # Random Vertical Flip
            if random.random() < 0.5:
                heightmap_tensor = TF.vflip(heightmap_tensor)
                tx_location_tensor = TF.vflip(tx_location_tensor)
                gain_tensor = TF.vflip(gain_tensor)
                if self.use_los_input:
                    los_tensor = TF.vflip(los_tensor)
                if self.use_pathloss_input:
                    pl_tensor = TF.vflip(pl_tensor)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.vflip(fhm_tensor)
                    
            k = random.randint(0, 3)
            if k > 0:
                heightmap_tensor = TF.rotate(heightmap_tensor, angle=k*90)
                tx_location_tensor = TF.rotate(tx_location_tensor, angle=k*90)
                gain_tensor = TF.rotate(gain_tensor, angle=k*90)
                if self.use_los_input:
                    los_tensor = TF.rotate(los_tensor, angle=k*90)
                if self.use_pathloss_input:
                    pl_tensor = TF.rotate(pl_tensor, angle=k*90)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.rotate(fhm_tensor, angle=k*90)
                    
            if random.random() < 0.5:
                heightmap_tensor = torch.transpose(heightmap_tensor, 1, 2)
                tx_location_tensor = torch.transpose(tx_location_tensor, 1, 2)
                gain_tensor = torch.transpose(gain_tensor, 1, 2)
                if self.use_los_input:
                    los_tensor = torch.transpose(los_tensor, 1, 2)
                if self.use_pathloss_input:
                    pl_tensor = torch.transpose(pl_tensor, 1, 2)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = torch.transpose(fhm_tensor, 1, 2)
                    
            if self.crop_resize:   
                shape_aug = transforms.RandomResizedCrop(
                    size=(256, 256),
                    scale=(0.1, 1.0),
                    ratio=(1, 1)
                )
                # i, j, h, w = shape_aug.get_params(img=heightmap_tensor, scale=shape_aug.scale, ratio=shape_aug.ratio)
                # heightmap_tensor = TF.resized_crop(heightmap_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                # tx_location_tensor = TF.resized_crop(tx_location_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                # gain_tensor = TF.resized_crop(gain_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                # if self.use_los_input:
                #     los_tensor = TF.resized_crop(los_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)

                i, j, h, w = shape_aug.get_params(img=heightmap_tensor, scale=shape_aug.scale, ratio=shape_aug.ratio)

                tx_coords = tx_location_tensor.nonzero(as_tuple=False)

                if tx_coords.numel() > 0:

                    tx_y = tx_coords[0, 1].item()
                    tx_x = tx_coords[0, 2].item()
                    
                    img_height, img_width = heightmap_tensor.shape[-2:]

                    top_min = max(0, tx_y - h + 1)
                    top_max = min(img_height - h, tx_y)

                    left_min = max(0, tx_x - w + 1)
                    left_max = min(img_width - w, tx_x)

                    if top_max >= top_min:
                        i = torch.randint(top_min, top_max + 1, size=(1,)).item()
                        
                    if left_max >= left_min:
                        j = torch.randint(left_min, left_max + 1, size=(1,)).item()

                heightmap_tensor = TF.resized_crop(heightmap_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                tx_location_tensor = TF.resized_crop(tx_location_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                gain_tensor = TF.resized_crop(gain_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                if self.use_los_input:
                    los_tensor = TF.resized_crop(los_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                if self.use_pathloss_input:
                    pl_tensor = TF.resized_crop(pl_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                if self.use_filt_and_norm and not self.filt_hm:
                    fhm_tensor = TF.resized_crop(fhm_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
                
        freq_tensor = torch.full_like(heightmap_tensor, float(self.freq))
        input_list = [heightmap_tensor, tx_location_tensor, freq_tensor]

        if self.use_filt_and_norm:
            input_list.append(fhm_tensor)
        if self.use_los_input:
            input_list.append(los_tensor)
        if self.use_pathloss_input:
            input_list.append(pl_tensor)

        inputs_tensor = torch.cat(input_list, dim=0)
            
            
        if self.transform:
            # Note: Standard torchvision transforms often expect PIL Image or (C,H,W) tensor.
            # If your custom transform expects something else, ensure compatibility.
            # The current setup provides inputs_tensor as (C, H, W) and gain_tensor as (1, H, W).
            inputs_tensor = self.transform(inputs_tensor) #.type(torch.float32) # type cast can be part of transform
            gain_tensor = self.transform(gain_tensor) #.type(torch.float32)

        # Ensure correct type after potential transform
        inputs_tensor = inputs_tensor.type(torch.float32)
        gain_tensor = gain_tensor.type(torch.float32)

        return (inputs_tensor, gain_tensor, base_name)
    
    def create_horizontal_line(self, array: np.ndarray, line_length: int) -> np.ndarray:
        new_array = array
        coords = np.where(new_array == 1)
        if coords[0].size == 0:
            print("Warning: No '1' found in the array.")
            return new_array
        row, col = coords[0][0], coords[1][0]
        half_length = (line_length - 1) // 2
        start_col = col - half_length
        end_col = col + half_length + 1
        array_width = new_array.shape[1]
        start_col_clipped = max(0, start_col)
        end_col_clipped = min(array_width, end_col)
        new_array[row, start_col_clipped:end_col_clipped] = 1
        return new_array

