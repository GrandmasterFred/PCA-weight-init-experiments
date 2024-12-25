# this stores all the utils that are needed in the experiments i guess

#region general tools
'''
==========================================
saving scripts and getting names 
==========================================
'''

import os
import shutil
import sys
from tqdm import tqdm
import psutil

def save_current_script_copy(folder_loc=None):
    '''
    Saves a copy of the currently running script.

    Parameters:
    - folder_loc (str): If None, saves script as script_name/script_name.py.
                        If a folder location is provided, saves as custom_folder/script_name.py.
    '''
    # Get the path of the currently running script
    current_script = os.path.abspath(sys.argv[0])
    
    # Extract the base name without the extension to create the folder
    base_name = os.path.basename(current_script)
    name, ext = os.path.splitext(base_name)
    
    if folder_loc is None:
        # Create the directory with the same name as the script (if it doesn't already exist)
        folder_name = name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    else:
        # Use the provided folder location
        folder_name = folder_loc
    
    # Define the destination path inside the new folder
    destination_path = os.path.join(folder_name, base_name)
    
    # Copy the current script to the new folder
    shutil.copy(current_script, destination_path)
    print(f"Copied current script to {destination_path}")

import os
def get_script_name_without_extension():
    # Get the path of the currently running script
    current_script = os.path.abspath(__file__)
    
    # Extract the base name without the extension
    name, _ = os.path.splitext(os.path.basename(current_script))
    
    return name


def create_sequential_folder():
    # Get the base folder name (script name without extension)
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    # Start with folder suffix 1
    counter = 1
    folder_name = f"{script_name}_{counter}"
    
    # Increment the counter until a non-existing folder is found
    while os.path.exists(folder_name):
        counter += 1
        folder_name = f"{script_name}_{counter}"
    
    # Create the folder
    os.makedirs(folder_name)
    
    return folder_name

#endregion

#region logging tools 

import logging
class MyLogger_modified_for_PCA:
    def __init__(self, log_file, log_level=logging.DEBUG):
        # setting up the logging location
        import os
        directory = os.path.dirname(log_file)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Configure logging settings
        self.log_file = log_file
        self.log_level = log_level
        logger = logging.getLogger()
        fhandler = logging.FileHandler(filename=log_file, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.DEBUG)

        # this should fix the issue of the PIL image stuff
        # https://github.com/camptocamp/pytest-odoo/issues/15 this should only allow warnings from PIL library to go through, while everythinng else is fine 
        logging.getLogger('PIL').setLevel(logging.WARNING)


    def log(self, message, level=logging.INFO):
        if level == logging.DEBUG:
            logging.debug(message)
        elif level == logging.INFO:
            logging.info(message)
        elif level == logging.WARNING:
            logging.warning(message)
        elif level == logging.ERROR:
            logging.error(message)
        elif level == logging.CRITICAL:
            logging.critical(message)
        else:
            raise ValueError("Invalid log level")



# we define a logging tool here that, when called, overrides the whole file so that it uses a built logger 
import builtins

# Save the original print function
original_print = print

def custom_print(logger, *args, **kwargs):
    # Log the message
    original_print(*args, **kwargs)
    logger.log(*args, **kwargs)

def override_print_with_logger(logger):
    def custom_print_with_logger(*args, **kwargs):
        custom_print(logger, *args, **kwargs)
    builtins.print = custom_print_with_logger


import psutil
import os
import subprocess
import torch

import psutil
import os
import subprocess
import torch

import psutil
import os
import subprocess
import torch

def get_memory_info():
    # 1. Get current amount of available RAM and total RAM
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / (1024 ** 3)  # Convert to GB
    total_ram = memory_info.total / (1024 ** 3)          # Convert to GB
    
    # Calculate the percentage of RAM used
    used_ram_percentage = memory_info.percent
    
    # 2. Get current amount of available and total VRAM using nvidia-smi
    try:
        # Fetching both free and total GPU memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        vram_info = result.stdout.decode('utf-8').strip().split('\n')[0].split(',')
        available_vram = float(vram_info[0].strip()) / 1024  # Convert to GB
        total_vram = float(vram_info[1].strip()) / 1024      # Convert to GB
        
        # Calculate used VRAM and its percentage
        used_vram = total_vram - available_vram
        used_vram_percentage = (used_vram / total_vram) * 100
    except Exception as e:
        available_vram = f"Error: {e}"
        total_vram = f"Error: {e}"
        used_vram_percentage = "Error"
    
    # 3. Get current amount of RAM used by the current Python script
    process = psutil.Process(os.getpid())
    used_ram_by_script = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    
    # 4. Get current amount of VRAM used by the current Python script (using PyTorch)
    if torch.cuda.is_available():
        used_vram_by_script = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    else:
        used_vram_by_script = "CUDA not available"
    
    return {
        'available_ram_gb': available_ram,
        'total_ram_gb': total_ram,
        'used_ram_percentage': used_ram_percentage,
        'available_vram_gb': available_vram,
        'total_vram_gb': total_vram,
        'used_vram_percentage': used_vram_percentage,
        'used_ram_by_script_gb': used_ram_by_script,
        'used_vram_by_script_gb': used_vram_by_script
    }

# # Example usage
# memory_info = get_memory_info()
# print(memory_info)


#endregion


#region dataloader compcars

'''this file does the custom development for the dataloader. this dataloader is intended to be used in situations where

1. the file is structured in file labels -> images
    Just like how compcars is structured
2. where we want the quick iteration of all the file locations 
    For cases where we do PCA sampling,
    For cases where we need to check if the data has around equal distribution with each other or not 

    
this dataloader will go through the sequence of
1. given a data location variable, get the list of all the images 
2. split that list into train, test, and val based on a random given seed/generator
3. check for whether
    all class exists 
    the distribution is around normal (meaning there is no class imbalance)
4. creates custom dataset that is returned

This custom dataset would have a special property of being able to
1. return the list of images that it is utilizing 
2. be used for a dataloader as well 
'''

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

# Custom PyTorch Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_list,transforms):
        """
        Args:
            image_list (list): List of tuples (image_path, label)
        """
        self.image_list = image_list
        self.transform = transforms     # this should be provided 
        self.image_list_only_loc = [loc for loc, _ in image_list]
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_path, label = self.image_list[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        subfolder = int(label) - 1       # this one converts it to subfolders, makes it easier 
        return image, subfolder
    

class CustomImageDataset_loadToMemory_loadtoVRAM(Dataset):
    def __init__(self, image_list, memcheck, transform=None):
        """
        Args:
            image_list (list): List of tuples (image_path, label)
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.

            this function loads all the image to vram, and keeps giving information on how much vram is still available
        """
        self.image_list = image_list
        self.transform = transform
        self.image_list_only_loc = [loc for loc, _ in image_list]
        self.images_transformed = False

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Estimate memory requirements
        raw_size, transformed_size = self.estimate_memory_req()

        # Load images into memory if possible
        memory_info = get_memory_info()
        print(memory_info)
        available_memory = memory_info['available_vram_gb'] # should be in gb
        print(f'estimated raw size {(raw_size/ (1024 ** 3)):.2f} estimated transformed size {(transformed_size/ (1024 ** 3)):.2f} available memory {(available_memory/ (1024 ** 3)):.2f}')
        if available_memory > (transformed_size/ (1024 ** 3)):
            self.image_tensor = self.load_images_to_memory_loadtoVRAM(apply_transform=True)
            self.images_transformed = True
        elif available_memory > raw_size:
            self.image_tensor = self.load_images_to_memory_loadtoVRAM(apply_transform=False)
        else:
            print("Warning: Insufficient memory to load images into memory. Images will be loaded from disk on demand.")
            self.image_tensor = None
        
        memory_info = get_memory_info()
        print(memory_info)

    def estimate_memory_req(self):
        """Estimate memory requirements for raw and transformed images."""
        sample_size = min(10, len(self.image_list))
        raw_size, transformed_size = 0, 0

        for i in range(sample_size):
            image = self.load_image(self.image_list[i][0])
            raw_tensor = transforms.ToTensor()(image)
            raw_size += raw_tensor.nelement() * raw_tensor.element_size()

            if self.transform:
                transformed_tensor = self.transform(image)
                # transformed_tensor = transforms.ToTensor()(transformed_image) # this is already a tensor 
                transformed_size += transformed_tensor.nelement() * transformed_tensor.element_size()
            else:
                transformed_size += raw_size

        # Scale estimates for the entire dataset
        scale_factor = len(self.image_list) / sample_size
        return raw_size * scale_factor, transformed_size * scale_factor

    def load_image(self, image_path):
        """Helper function to load and convert an image to RGB."""
        return Image.open(image_path).convert('RGB')

    def load_images_to_memory_loadtoVRAM(self, apply_transform):
        """Load images into memory, applying the transform if requested."""
        image_tensors = []
        used_memory = 0

        for image_path, _ in self.image_list:
            image = self.load_image(image_path)
            
            if apply_transform and self.transform:
                tensor = self.transform(image)
            else:
                tensor = torch.tensor(image)
            
            tensor = tensor.to(self.device)
            image_tensors.append(tensor)

        return torch.stack(image_tensors) if image_tensors else None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.image_tensor is not None:
            image = self.image_tensor[idx]
            if not self.images_transformed and self.transform:
                image = self.transform(image)
        else:
            # this means that it was not pre-loaded for some reason
            image = self.load_image(self.image_list[idx][0])
            if self.transform:
                image = self.transform(image)
            try:
                image = transforms.ToTensor()(image)
            except Exception as e:
                # this is just to transform the image to tensor if it is not a tensor i guess 
                dummy_var = 0

        label = self.image_list[idx][1]
        subfolder = int(label) -1
        return image, subfolder


class CustomImageDataset_loadToMemory(Dataset):
    def __init__(self, image_list, transform=None):
        """
        Args:
            image_list (list): List of tuples (image_path, label)
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        """
        self.image_list = image_list
        self.transform = transform
        self.image_list_only_loc = [loc for loc, _ in image_list]
        self.images_transformed = False

        # Estimate memory requirements
        raw_size, transformed_size = self.estimate_memory_req()

        # Load images into memory if possible
        available_memory = psutil.virtual_memory().available
        print(f'estimated raw size {(raw_size/ (1024 ** 3)):.2f} estimated transformed size {(transformed_size/ (1024 ** 3)):.2f} available memory {(available_memory/ (1024 ** 3)):.2f}')
        if available_memory > transformed_size:
            self.image_tensor = self.load_images_to_memory(apply_transform=True)
            self.images_transformed = True
        elif available_memory > raw_size:
            self.image_tensor = self.load_images_to_memory(apply_transform=False)
        else:
            print("Warning: Insufficient memory to load images into memory. Images will be loaded from disk on demand.")
            self.image_tensor = None

    def estimate_memory_req(self):
        """Estimate memory requirements for raw and transformed images."""
        sample_size = min(10, len(self.image_list))
        raw_size, transformed_size = 0, 0

        for i in range(sample_size):
            image = self.load_image(self.image_list[i][0])
            raw_tensor = transforms.ToTensor()(image)
            raw_size += raw_tensor.nelement() * raw_tensor.element_size()

            if self.transform:
                transformed_tensor = self.transform(image)
                # transformed_tensor = transforms.ToTensor()(transformed_image) # this is already a tensor 
                transformed_size += transformed_tensor.nelement() * transformed_tensor.element_size()
            else:
                transformed_size += raw_size

        # Scale estimates for the entire dataset
        scale_factor = len(self.image_list) / sample_size
        return raw_size * scale_factor, transformed_size * scale_factor

    def load_image(self, image_path):
        """Helper function to load and convert an image to RGB."""
        return Image.open(image_path).convert('RGB')

    def load_images_to_memory(self, apply_transform):
        """Load images into memory, applying the transform if requested."""
        image_tensors = []
        used_memory = 0
        available_memory = psutil.virtual_memory().available

        for image_path, _ in self.image_list:
            image = self.load_image(image_path)
            if apply_transform and self.transform:
                tensor = self.transform(image)
            # tensor = transforms.ToTensor()(image) already in tensor 
            used_memory += tensor.nelement() * tensor.element_size()

            if used_memory > available_memory * 0.9:
                print("Warning: Approaching memory limit while loading images.")
                return None

            image_tensors.append(tensor)

        # this prints out the used memory for reference
        print(f'the total memory used for this is {(used_memory/ (1024 ** 3)):.2f} out {(available_memory/ (1024 ** 3)):.2f}')
        return torch.stack(image_tensors) if image_tensors else None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.image_tensor is not None:
            image = self.image_tensor[idx]
            if not self.images_transformed and self.transform:
                image = self.transform(image)
        else:
            image = self.load_image(self.image_list[idx][0])
            if self.transform:
                image = self.transform(image)
            image = transforms.ToTensor()(image)

        label = self.image_list[idx][1]
        subfolder = int(label) -1
        return image, subfolder

# Function to split data into train, val, test sets
# Function to split data into train, val, test sets using stratified sampling
def split_data(data_loc, seed):
    """
    Args:
        data_loc (str): Root directory containing the data
        seed (int): Random seed for reproducibility
    Returns:
        train_set, val_set, test_set (list): Lists of (image_path, label) for each set
    """
    image_list = []
    
    # Get all images in data_loc and their labels
    for root, _, files in os.walk(data_loc):
        label = os.path.basename(root)
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                image_list.append((os.path.join(root, file), label))
    
    # Extract labels for stratification
    labels = [label for _, label in image_list]

    if not labels:
        raise ValueError("The label list is empty. No images were found in the provided directory.")
    
    # Split the data into training and temp sets (temp set will be split into val and test)
    train_set, temp_set, train_labels, temp_labels = train_test_split(
        image_list, labels, test_size=0.2, stratify=labels, random_state=seed)
    
    # Split the temp set into validation and test sets
    val_set, test_set, val_labels, test_labels = train_test_split(
        temp_set, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed)
    
    print(f'Total images: {len(image_list)}')
    print(f'Train length: {len(train_set)}, Val length: {len(val_set)}, Test length: {len(test_set)}')
    
    return train_set, val_set, test_set

# Function to validate the label distribution and coverage

def validate_sets(train_set, val_set, test_set):
    """
    Args:
        train_set, val_set, test_set (list): Lists of (image_path, label) for each set
    Returns:
        bool: True if all checks pass, False otherwise
    """
    # Extract labels from each set
    train_labels = [label for _, label in train_set]
    val_labels = [label for _, label in val_set]
    test_labels = [label for _, label in test_set]
    
    # 1) Check if all labels in the train set appear in the val and test sets
    unique_train_labels = set(train_labels)
    unique_val_labels = set(val_labels)
    unique_test_labels = set(test_labels)
    
    if not unique_train_labels.issubset(unique_val_labels) or not unique_train_labels.issubset(unique_test_labels):
        print("Validation Failed: Not all labels in train set appear in val and test sets.")
        return False
    
    # 2) Check if the distribution of the three sets is approximately equal using the K-S test
    
    # Perform a K-S test for distribution similarity between train and val
    ks_stat_train_val, p_val_train_val = ks_2samp(train_labels, val_labels)
    
    # Perform a K-S test for distribution similarity between train and test
    ks_stat_train_test, p_val_train_test = ks_2samp(train_labels, test_labels)
    
    if p_val_train_val < 0.05 or p_val_train_test < 0.05:
        print("Validation Failed: Label distribution is significantly different across sets.")
        return False
    
    print("Validation Passed.")
    return True

# Function to create datasets from the lists
def create_datasets(train_set, val_set, test_set,transforms):
    """
    Args:
        train_set, val_set, test_set (list): Lists of (image_path, label) for each set
    Returns:
        train_dataset, val_dataset, test_dataset (Dataset): PyTorch datasets for each set
    """
    train_dataset = CustomImageDataset_loadToMemory(train_set,transforms)
    val_dataset = CustomImageDataset_loadToMemory(val_set,transforms)
    test_dataset = CustomImageDataset_loadToMemory(test_set,transforms)
    
    return train_dataset, val_dataset, test_dataset

def create_datasets_loadtoVRAM(train_set, val_set, test_set,transforms, memcheck):
    """
    Args:
        train_set, val_set, test_set (list): Lists of (image_path, label) for each set
    Returns:
        train_dataset, val_dataset, test_dataset (Dataset): PyTorch datasets for each set
    """
    train_dataset = CustomImageDataset_loadToMemory_loadtoVRAM(train_set, memcheck,transforms)
    val_dataset = CustomImageDataset_loadToMemory_loadtoVRAM(val_set, memcheck,transforms)
    test_dataset = CustomImageDataset_loadToMemory_loadtoVRAM(test_set, memcheck,transforms)
    
    return train_dataset, val_dataset, test_dataset

def create_datasets_DONTLOADMEMORY(train_set, val_set, test_set,transforms):
    """
    Args:
        train_set, val_set, test_set (list): Lists of (image_path, label) for each set
    Returns:
        train_dataset, val_dataset, test_dataset (Dataset): PyTorch datasets for each set
    """
    train_dataset = CustomImageDataset(train_set,transforms)
    val_dataset = CustomImageDataset(val_set,transforms)
    test_dataset = CustomImageDataset(test_set,transforms)
    
    return train_dataset, val_dataset, test_dataset

def create_datasets_DONTLOADTRAIN(train_set, val_set, test_set,transforms):
    """
    Args:
        train_set, val_set, test_set (list): Lists of (image_path, label) for each set
    Returns:
        train_dataset, val_dataset, test_dataset (Dataset): PyTorch datasets for each set
    """
    train_dataset = CustomImageDataset(train_set,transforms)
    val_dataset = CustomImageDataset(val_set,transforms)
    test_dataset = CustomImageDataset_loadToMemory(test_set,transforms)
    
    return train_dataset, val_dataset, test_dataset

# # Example usage
# if __name__ == "__main__":
#     data_location = 'E:\compcars\compCars_sv_modified_enhanced'

#     # setting up a custom transform to test it out 
#     transform = transforms.Compose([
#             transforms.Resize((299, 299)),  # InceptionV3 input size
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])
    
#     # looking for a random seed that satisfies critiria
#     correct_dist = False
#     while correct_dist is not True:
#         seed = random.randint(0, 9999)

#         train_set, val_set, test_set = split_data(data_location, seed)
    
#         if validate_sets(train_set, val_set, test_set):
#             train_dataset, val_dataset, test_dataset = create_datasets(train_set, val_set, test_set, transforms= transform)
#             print(f"Datasets created: Train size = {len(train_dataset)}, Val size = {len(val_dataset)}, Test size = {len(test_dataset)}")
#             correct_dist = True
#             print(f'seed {seed} is valid')
#         else:
#             # this means that the set is not validated, and this will run again
#             print(f'seed {seed} is not valid')
    
#     # this section tests the functionality of getting the image list from it 
#     test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     # the bottom enumeration looks correct enough
#     for idx, (img, label) in enumerate(test_loader):
#         print(f'shape: {img.shape()} label: {label.shape()}')
#         print(f'{label}')
#         break


# small neat function that accomplishes all that 
def custom_dataset_compcars(root_dir, transform, seed=None):
    # this function will be called from the file. 
    '''
    This function accepts: 
        root dir, transform, seed
    Returns:
        train_dataset, val_dataset, test_dataset
    '''
    
    # looking for a random seed that satisfies critiria
    correct_dist = False
    while correct_dist is not True:
        if seed is None:
            seed = random.randint(0, 9999)

        train_set, val_set, test_set = split_data(root_dir, seed)
    
        if validate_sets(train_set, val_set, test_set):
            train_dataset, val_dataset, test_dataset = create_datasets(train_set, val_set, test_set, transforms= transform)
            print(f"Datasets created: Train size = {len(train_dataset)}, Val size = {len(val_dataset)}, Test size = {len(test_dataset)}")
            correct_dist = True
            print(f'seed {seed} is valid')
        else:
            # this means that the set is not validated, and this will run again
            print(f'seed {seed} is not valid')
    
    return train_dataset, val_dataset, test_dataset

def custom_dataset_compcars_DONTLOADMEMORY(root_dir, transform, seed=None):
    # this function will be called from the file. 
    '''
    This function accepts: 
        root dir, transform, seed
    Returns:
        train_dataset, val_dataset, test_dataset
    '''
    
    # looking for a random seed that satisfies critiria
    correct_dist = False
    while correct_dist is not True:
        if seed is None:
            seed = random.randint(0, 9999)

        train_set, val_set, test_set = split_data(root_dir, seed)
    
        if validate_sets(train_set, val_set, test_set):
            train_dataset, val_dataset, test_dataset = create_datasets_DONTLOADMEMORY(train_set, val_set, test_set, transforms= transform)
            print(f"Datasets created: Train size = {len(train_dataset)}, Val size = {len(val_dataset)}, Test size = {len(test_dataset)}")
            correct_dist = True
            print(f'seed {seed} is valid')
        else:
            # this means that the set is not validated, and this will run again
            print(f'seed {seed} is not valid')
    
    return train_dataset, val_dataset, test_dataset

def custom_dataset_compcars_DONTLOADTRAIN(root_dir, transform, seed=None):
    # this function will be called from the file. 
    '''
    This function accepts: 
        root dir, transform, seed
    Returns:
        train_dataset, val_dataset, test_dataset
    '''
    
    # looking for a random seed that satisfies critiria
    correct_dist = False
    while correct_dist is not True:
        if seed is None:
            seed = random.randint(0, 9999)

        train_set, val_set, test_set = split_data(root_dir, seed)
    
        if validate_sets(train_set, val_set, test_set):
            train_dataset, val_dataset, test_dataset = create_datasets_DONTLOADTRAIN(train_set, val_set, test_set, transforms= transform)
            print(f"Datasets created: Train size = {len(train_dataset)}, Val size = {len(val_dataset)}, Test size = {len(test_dataset)}")
            correct_dist = True
            print(f'seed {seed} is valid')
        else:
            # this means that the set is not validated, and this will run again
            print(f'seed {seed} is not valid')
    
    return train_dataset, val_dataset, test_dataset


def custom_dataset_compcars_loadtoVRAM(root_dir, transform, seed=None, memcheck=False):
    # this function will be called from the file. 
    '''
    This function accepts: 
        root dir, transform, seed
    Returns:
        train_dataset, val_dataset, test_dataset
    '''
    
    # looking for a random seed that satisfies critiria
    correct_dist = False
    while correct_dist is not True:
        if seed is None:
            seed = random.randint(0, 9999)

        train_set, val_set, test_set = split_data(root_dir, seed)
    
        if validate_sets(train_set, val_set, test_set):
            train_dataset, val_dataset, test_dataset = create_datasets_loadtoVRAM(train_set, val_set, test_set, transforms= transform, memcheck=memcheck)
            print(f"Datasets created: Train size = {len(train_dataset)}, Val size = {len(val_dataset)}, Test size = {len(test_dataset)}")
            correct_dist = True
            print(f'seed {seed} is valid')
        else:
            # this means that the set is not validated, and this will run again
            print(f'seed {seed} is not valid')
    
    return train_dataset, val_dataset, test_dataset


#endregion




