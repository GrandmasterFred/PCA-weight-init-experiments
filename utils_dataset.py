import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from collections import defaultdict
import os 
import random
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

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
    train_dataset = CustomImageDataset(train_set,transforms)
    val_dataset = CustomImageDataset(val_set,transforms)
    test_dataset = CustomImageDataset(test_set,transforms)
    
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



def create_dataloader_compcars(data_loc, transform, batch_size, gen, split_ratio=0.8, check_dist=False):
    # this is the old version of dataloading, i am now using the one in dev_files\custom_dataloader_dev.py
    # Create the dataset using ImageFolder
    dataset = torchvision.datasets.ImageFolder(root=data_loc, transform=transform)
    
    # Split dataset into train and validation datasets
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=gen)
    
    # Split validation dataset into validation and test datasets
    test_size = int(0.5 * len(val_dataset))
    val_size = len(val_dataset) - test_size
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [val_size, test_size], generator=gen)
    
    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print('Dataset loading successful')
    
    # Print out the number of samples in each DataLoader
    print(f'Number of samples - Train: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
    
    if check_dist:
        # Function to count label occurrences in a dataloader using torch.Tensor.bincount
        def count_labels(dataloader):
            all_labels = []
            for _, labels in tqdm(dataloader):
                all_labels.append(labels)
            # Concatenate all labels and count occurrences
            all_labels = torch.cat(all_labels)
            label_counts = torch.bincount(all_labels)
            return label_counts
        
        # Count label occurrences in train, val, and test loaders
        # print(f'train count')
        # train_label_counts = count_labels(train_loader)
        print(f'val count')
        val_label_counts = count_labels(val_loader)
        print(f'test count')
        test_label_counts = count_labels(test_loader)
        
        # Convert counts to dictionaries for easy printing
        # train_label_counts_dict = {i: count for i, count in enumerate(train_label_counts)}
        val_label_counts_dict = {i: count for i, count in enumerate(val_label_counts)}
        test_label_counts_dict = {i: count for i, count in enumerate(test_label_counts)}
        
        # Print label distributions
        # print(f'Train label distribution: {train_label_counts_dict}')
        # print(f'Validation label distribution: {val_label_counts_dict}')
        # print(f'Test label distribution: {test_label_counts_dict}')
        
        # Get unique labels in each set
        # train_labels = set(train_label_counts_dict.keys())
        val_labels = set(val_label_counts_dict.keys())
        test_labels = set(test_label_counts_dict.keys())

        print(f'val length: {len(val_labels)} test len: {len(test_labels)}')
        
        # # Find labels in train set that are missing in validation or test sets
        # missing_in_val = train_labels - val_labels
        # missing_in_test = train_labels - test_labels
        
        # if missing_in_val:
        #     print(f"Warning: The following labels are missing in the validation set: {missing_in_val}")
        # if missing_in_test:
        #     print(f"Warning: The following labels are missing in the test set: {missing_in_test}")
    
    return train_loader, val_loader, test_loader

def create_pca_dataloader(train_list, samples_per_label, transform, batch_size, num_classes):
    """
    Args:
        train_list (list): Dataset containing image paths only, intended for the training set 
        samples_per_label (int): Number of samples to collect per label.
        transform (callable): Transformation to apply to the images.
        batch_size (int): Batch size for the DataLoader.
        num_classes (int): Number of classes.
    Returns:
        DataLoader: DataLoader for the sampled PCA images.
    """
    train_img_loc_for_pca_samples = train_list

    # Dictionary to hold images grouped by their labels
    label_to_images = defaultdict(list)

    # Group images by their labels
    for image_path in train_img_loc_for_pca_samples:
        label = os.path.basename(os.path.dirname(image_path))  # Extract the label from the path
        label_to_images[label].append(image_path)

    # List to hold the sampled images
    sampled_list = []
    fewer_imgs_than_we_need_counter = 0

    # Randomly sample images for each label
    for label, images in label_to_images.items():
        if len(images) >= samples_per_label:
            sampled_images = random.sample(images, samples_per_label)
        else:
            sampled_images = images  # If fewer images than samples_per_label, take all
            fewer_imgs_than_we_need_counter += 1
        sampled_list.extend(sampled_images)

    if fewer_imgs_than_we_need_counter > 0:
        print(f'There are {fewer_imgs_than_we_need_counter} classes that do not meet PCA sampling requirements.')

    print(f'We sampled {len(sampled_list)} from {len(train_img_loc_for_pca_samples)} samples, at {samples_per_label} samples for {num_classes} classes.')

    # Convert the sampled list into a tuple of (image_path, label)
    sampled_training_images_for_pca_tuple = [
        (item, os.path.basename(os.path.dirname(item))) for item in sampled_list
    ]

    # Create the dataset using the CustomImageDataset class
    samples_for_pca_dataset = CustomImageDataset(
        image_list=sampled_training_images_for_pca_tuple,
        transforms=transform
    )

    # Create the DataLoader
    samples_for_pca_dataloader = DataLoader(
        samples_for_pca_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return samples_for_pca_dataloader


def create_pca_dataloader_from_imgloc_n_label_list(train_list, samples_per_label, transform, batch_size, num_classes):
    """
    Args:
        train_list (list): Dataset containing tuple of image path and image label  
        samples_per_label (int): Number of samples to collect per label.
        transform (callable): Transformation to apply to the images.
        batch_size (int): Batch size for the DataLoader.
        num_classes (int): Number of classes.
    Returns:
        DataLoader: DataLoader for the sampled PCA images.
    """
    # using train_list, we randomly sample based on second part of the tuple 
    # we first determine how many classes there are 
    label_to_paths = defaultdict(list)

    for path, label in train_list:
        label_to_paths[label].append(path)
    
    def sample_n_from_labels(label_to_paths, N):
        sampled_data = []
        
        for label, paths in label_to_paths.items():
            # Make sure we do not sample more than available paths
            sample_size = min(N, len(paths))
            sampled_paths = random.sample(paths, sample_size)
            
            # Add the sampled (path, label) tuples to the list
            sampled_data.extend([(path, label) for path in sampled_paths])
        
        return sampled_data
    
    sampled_list = sample_n_from_labels(label_to_paths, samples_per_label)

    print(f'We sampled {len(sampled_list)} from {len(train_list)} samples, at {samples_per_label} samples for {num_classes} classes.')

    # Convert the sampled list into a tuple of (image_path, label)
    sampled_training_images_for_pca_tuple = sampled_list
    # modify this so that it fits custom image dataset, since that one expexts labels 1-n, whereas this is encoded in 0-n

    # Create the dataset using the CustomImageDataset class
    samples_for_pca_dataset = CustomImageDataset(
        image_list=sampled_training_images_for_pca_tuple,
        transforms=transform
    )

    # Create the DataLoader
    samples_for_pca_dataloader = DataLoader(
        samples_for_pca_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return samples_for_pca_dataloader