# this file will load and train a basic resnet18 implementation. This will be where we test out the vram size and how much we can shove on it while maintaining good training stuff 
# to make it faster, we will be referencing D:\gitprojects\PCA_paper_exp_data\load_to_gpu_DEV.py

import torch
import random
import os
import torchvision.transforms as transforms
from utils_PCA_paper_exp_data import MyLogger_modified_for_PCA, create_sequential_folder
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

#region var
RANDOM_SEED= random.randint(1000, 9999)
DATA_LOC = '/ibm/gpfs/home/kpha0008/data/E11_fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/'
DATA_NAMES = 'full'
TRAIN_FROM_SCRATCH = True
DETECT_ANON = False
MODEL_NAME = create_sequential_folder()    # this would just be the file name minus the extension
# everything would be saved based on this model name 

PCA_SAMPLES_PER_CLASS = 2
NUM_CLASSES = 30
IMPLEMENT_PCA = True
PCA_LAYERS = 4      # this is currently useless

BATCH_SIZE = 32
CHECK_DIST = False   # this simply checks if all the classes exists for the val and test set 

# for this, it takes prio in order of pca -> trans -> random int. Meaning if random int is needed, both pca and trans has to be false 
INITIALIZE_PCA = True
INITIALZE_TRANS_WEIGHTS = False

# this is for the arg dict
LR = 0.0024         # slightly lowered due to bumped up batch size 
MAX_EPOCH = 150
IDLE_EPOCH = 10
OUTPUT_NAME = MODEL_NAME
# the optimizer and the criterion is at the argdict


# region supporting func

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



# endregion

#region aircraft dataset 

class e11_aircraft_load_to_vram(Dataset):
    def __init__(self, image_list, manufac_path, correlation_list, transform=None):
        """
        Args:
            image_list (list): List of tuples (image_path, label)
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.

            this function loads all the image to vram, and keeps giving information on how much vram is still available
        """
        self.manufac_path = manufac_path            # text file that provides a list of manufac
        self.correlation_list = correlation_list    # text file that provides a linkage between image name and manufac
        self.image_list = image_list
        self.transform = transform
        self.image_list_only_loc = [loc for loc, _ in image_list]
        self.images_transformed = False

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # setup self.image_loc_family_list
        self.e11_aircraft_family_getting()     

        # modify the image_list so that it uses self.image_loc_family_list, so that the loading goes correctly
        self.image_list = self.image_loc_family_list

        # this is for passing to the PCA function, would make out life easier 
        # it adds 1 to the label since that is what the custom dataset expects
        self.image_loc_label = []
        for image_loc, manufac_name in self.image_loc_family_list:
            self.image_loc_label.append((image_loc, self.label_dict[manufac_name] + 1))

        memory_info = get_memory_info()
        print(memory_info)

        # this section tries to load stuff into ram instead, and place it as a list of PIL images 
        try:
            self.list_of_pil_images = self.load_images_to_ram()
        except Exception as e:
            print(f'failed to load all images or something')
            print(e)
        
        memory_info = get_memory_info()
        print(f'this is after loading images to ram {memory_info}')

    def load_images_to_ram(self):
        """This function uses self.image_loc_family_list to try to load images to memory 
        """
        list_of_pil_images = []

        for loc, _ in self.image_loc_family_list:
            list_of_pil_images.append(self.load_image(loc))
        return list_of_pil_images
    
    
    def generate_label_dict(self, file_path):
        """Reads a text file and generates a dictionary mapping each name to an integer label."""
        label_dict = {}
        
        # Open and read the file line by line
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                # Strip any whitespace characters and add to dictionary with label starting from 0
                name = line.strip()
                label_dict[name] = idx
        
        return label_dict
    
    def read_file_and_create_list(self, filename):
        result = []
        
        with open(filename, 'r') as file:
            for line in file:
                # Strip any extra whitespace and split the line by the first whitespace
                parts = line.strip().split(maxsplit=1)
                
                # Ensure there are exactly two parts (text1, text2)
                if len(parts) == 2:
                    text1, text2 = parts
                    result.append((text1, text2))
        
        return result


    def e11_aircraft_family_getting(self):
        # this function takes in the list of locations [(image loc, root)] and then returns a list of lcoations with the family instead. This also defines a dictionary that links label -> class name 

        # this will be the storage list 
        self.image_loc_family_list = []

        # this one generates the dictionary from the label file 
        self.label_dict = self.generate_label_dict(self.manufac_path)

        # this one loads a text file, and reads it line by line into a list. 
        matching_list = self.read_file_and_create_list(self.correlation_list)


        for (location, _) in self.image_list:
            # from the location variable, get the basename without any of the .jpg extension, name this var_match
            var_match = os.path.splitext(os.path.basename(location))[0]

            # look through matching_list, which is a list of tuples, where the tuples are arranged as (var, manufacturer)
            # find the element in the list where var == var_match, and note down the manufacturer in temp_manufac
            temp_manufac = None
            for var, manufacturer in matching_list:
                if var == var_match:
                    temp_manufac = manufacturer
                    break  # Once we find the match, we can exit the loop early
            
            if temp_manufac is None:
                # Optional: Handle the case where no manufacturer is found, such as setting a default or skipping
                temp_manufac = "Unknown"  # Or any other fallback mechanism

            # create tuple of (location, temp_manufac) and append it to list self.image_loc_family_list
            self.image_loc_family_list.append((location, temp_manufac))
    
    def estimate_memory_req(self):
        """Estimate memory requirements for raw image files on disk."""
        total_size = 0

        # Loop over all image paths in the dataset
        for image_path in self.image_list:
            # Get the file size of the image in bytes
            total_size += os.path.getsize(image_path[0])

        return total_size       # this is in bytes 

    def load_image(self, image_path):
        """Helper function to load and convert an image to RGB."""
        return Image.open(image_path).convert('RGB')

    def load_images_to_memory_loadtoVRAM(self):
        """Load images into VRAM (CUDA) without transforming them."""
        image_tensors = []

        for image_path, _ in self.image_list:
            try:
                # Load the image
                image = self.load_image(image_path)
                # Convert the image to a tensor without applying transformations
                raw_tensor = transforms.ToTensor()(image)
                # Move the raw tensor to the CUDA device (VRAM)
                raw_tensor = raw_tensor.to(self.device)
                # Append to the list
                image_tensors.append(raw_tensor)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")

        return image_tensors if image_tensors else None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # in this case, the image is saved in a list, where the list is root, image 

        # location 
        _, manufac = self.image_loc_family_list[idx]

        # get the image, which is already saved self.image_tensors
        image = self.list_of_pil_images[idx]    # this should be in PIL

        # transform the pil image
        tensor = self.transform(image)
        
        # convert the manufac name into label using dict. The dict label starts from 0 already, so no need to -1
        label = self.label_dict[manufac]

        return tensor, label     # where subfolder starts from 0


def create_datasets_loadtoVRAM(train_set, val_set, test_set,transforms, manufac_path, correlation_list):
    """
    Args:
        train_set, val_set, test_set (list): Lists of (image_path, label) for each set
    Returns:
        train_dataset, val_dataset, test_dataset (Dataset): PyTorch datasets for each set
    """
    train_dataset = e11_aircraft_load_to_vram(train_set, manufac_path, correlation_list,transforms)
    val_dataset = e11_aircraft_load_to_vram(val_set, manufac_path, correlation_list,transforms)
    test_dataset = e11_aircraft_load_to_vram(test_set, manufac_path, correlation_list,transforms)
    
    return train_dataset, val_dataset, test_dataset


def e11_aircraft_dataset(root_dir, transform, seed, manufac_path, correlation_list):
    '''
    this function accepts:
        root dir, transform, and memory check
    this function returns:
        tran_dataset, val_dataset, and test_dataset 
    '''

    # splits the dataset into train_set, val_set, and test_set, this process already splits it perfectly, so i dont have to validate it 
    train_set, val_set, test_set = split_data(root_dir, seed)

    # create the dataset, and load it to vram. This dataset only stores RAW images, and has not have them transformed yet. Raw images will be in tensor foramt 
    train_dataset, val_dataset, test_dataset = create_datasets_loadtoVRAM(train_set, val_set, test_set, transforms= transform, manufac_path=manufac_path, correlation_list = correlation_list)

    # returns the datasets
    return train_dataset, val_dataset, test_dataset


# this section tests the dataset 

# this is specific for the airline one, since 20 pixels needed to be removed 
class e11_RemoveBottomRows(object):
    """Remove the bottom 20 rows of a PIL image."""

    def __init__(self, num_rows_to_remove=20):
        self.num_rows_to_remove = num_rows_to_remove

    def __call__(self, img):
        # Ensure the input is a PIL image
        if not isinstance(img, Image.Image):
            raise TypeError(f"Input should be a PIL image, got {type(img)}")

        # Get image dimensions
        width, height = img.size

        # Define new cropped box (left, upper, right, lower)
        crop_box = (0, 0, width, height - self.num_rows_to_remove)

        # Crop the image to remove the bottom rows
        img = img.crop(crop_box)

        return img


# this section tests the dataset 

# this is specific for the airline one, since 20 pixels needed to be removed 
class e11_RemoveBottomRows(object):
    """Remove the bottom 20 rows of a PIL image."""

    def __init__(self, num_rows_to_remove=20):
        self.num_rows_to_remove = num_rows_to_remove

    def __call__(self, img):
        # Ensure the input is a PIL image
        if not isinstance(img, Image.Image):
            raise TypeError(f"Input should be a PIL image, got {type(img)}")

        # Get image dimensions
        width, height = img.size

        # Define new cropped box (left, upper, right, lower)
        crop_box = (0, 0, width, height - self.num_rows_to_remove)

        # Crop the image to remove the bottom rows
        img = img.crop(crop_box)

        return img
# endregion



gen = torch.Generator()
gen.manual_seed(RANDOM_SEED)

loggerName = MODEL_NAME + '.log'
loggerName = os.path.join(MODEL_NAME, loggerName)
logger = MyLogger_modified_for_PCA(loggerName)

logger.log(f'random seed used in this iteration is {RANDOM_SEED}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# saving a copy for reproduction
from utils_PCA_paper_exp_data import save_current_script_copy
save_current_script_copy(folder_loc=MODEL_NAME)

# setting up so that any print function in this script also uses the logger 
from utils_PCA_paper_exp_data import override_print_with_logger
override_print_with_logger(logger)

#endregion

#region dataloader 
transform = transforms.Compose([
            e11_RemoveBottomRows(num_rows_to_remove=20),  # Remove the bottom 20 rows first
            transforms.Resize((224, 224)),  # InceptionV3 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for single channel
        ])

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to InceptionV3 input size
#     transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize for single grayscale channel
# ])

from utils_PCA_paper_exp_data import get_memory_info
memory_info = get_memory_info()
print(f'before loading dataset {memory_info}')

aircraft_root = DATA_LOC
aircraft_images = os.path.join(aircraft_root, 'images')
aircraft_manufac = os.path.join(aircraft_root, 'manufacturers.txt')
aircraft_correlation_list = os.path.join(aircraft_root, 'images_manufacturer_all.txt')

train_dataset, val_dataset, test_dataset = e11_aircraft_dataset(
    root_dir = aircraft_images, 
    transform = transform, 
    seed=RANDOM_SEED,
    manufac_path = aircraft_manufac,
    correlation_list = aircraft_correlation_list
    )

memory_info = get_memory_info()
print(f'after loading dataset {memory_info}')

# create dataloaders from those datasets 
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#endregion

# region network int 
import torch
import torchvision.models as models
model = models.vgg11(pretrained=False)

# model.features[0] = nn.Conv2d(in_channels=1, 
#                               out_channels=64, 
#                               kernel_size=(3, 3), 
#                               stride=(1, 1), 
#                               padding=(1, 1))

# Ensure the model is in evaluation mode
model.eval()
model.to(device)

# endregion 

# region PCA int 
activation = {} # stores in dict 
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

from torchvision.models.resnet import BasicBlock, ResNet

def get_conv_layers_with_prev_vgg11(model):
    conv_layers = []
    previous_layer = None

    first_iter = True

    for name, layer in model.named_modules():
        # If it's a Conv2d layer, store it along with the preceding layer
        if isinstance(layer, nn.Conv2d):
            if previous_layer is None:
                conv_layers.append((layer, None, name))
            else:
                conv_layers.append((layer, previous_layer, name))
        
        if not isinstance(layer, (nn.Sequential, nn.ModuleList, BasicBlock, ResNet)):
            if not first_iter:      # this is just because of a bug on specifically the first layer
                previous_layer = layer
            else:
                previous_layer = None
                first_iter = False

    return conv_layers

def return_percentage_of_list(lst, percentage):
    # Ensure percentage is between 0 and 100
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")

    # Calculate how many elements correspond to the given percentage
    length = len(lst)
    num_elements = int(length * (percentage / 100.0))

    # Return the corresponding portion of the list
    return lst[:num_elements]

conv_layers_with_required_fmaps = get_conv_layers_with_prev_vgg11(model)
conv_layers_with_required_fmaps = return_percentage_of_list(conv_layers_with_required_fmaps, 50)        # this gets only 50% i guess


from utils_dataset import create_pca_dataloader_from_imgloc_n_label_list
# this should only accept lists that encodes 
pca_sample_loader = create_pca_dataloader_from_imgloc_n_label_list(
    train_list=train_dataset.image_loc_label,
    samples_per_label=PCA_SAMPLES_PER_CLASS,
    transform=transform,
    num_classes=NUM_CLASSES,
    batch_size=32       # defining a custom batch size here, idk how much the model can handle
)

feature_map = None

from tqdm import tqdm
# gathering the first set of feature maps for the pca 
for idx, (data, label) in enumerate(tqdm(pca_sample_loader)):
    data = data.to(device)

    if feature_map is None:
        # this one just assumes the other feature maps are none as well
        feature_map = data.clone()
    else:
        # meaning the map is not empty, and as such can be concatenated
        feature_map = torch.cat((feature_map, data.clone()), dim=0)

# looping through all of those to initialize for PCA 
from utilsPCA import generate_and_set_pca_broadcast_equal_int

for (target_conv, target_output, target_name) in conv_layers_with_required_fmaps:
    print(f'for: {target_conv} \n we are using: {target_output} target name: {target_name}')
    
    # setting the feature map to be none at the start, so everything is collected fresh 
    feature_map = None

    # gathers the feature maps. If there is no targeted pre-layer, we use raw images directly 
    if target_output is not None:
        target_output.register_forward_hook(get_activation(target_name))

        # gathering the first set of feature maps for the pca 
        for idx, (data, label) in enumerate(tqdm(pca_sample_loader)):
            data = data.to(device)

            # send the data through, then extract the data 
            _ = model(data)         # we dont actually need to keep the output

            fmap_temp = activation[target_name]     # already detached

            if feature_map is None:
                # this one just assumes the other feature maps are none as well
                feature_map = fmap_temp
            else:
                # meaning the map is not empty, and as such can be concatenated
                feature_map = torch.cat((feature_map, fmap_temp), dim=0)

    else:
        print(f'using raw input directly')

        for idx, (data, label) in enumerate(tqdm(pca_sample_loader)):
            data = data.to(device)

            if feature_map is None:
                # this one just assumes the other feature maps are none as well
                feature_map = data.clone()
            else:
                # meaning the map is not empty, and as such can be concatenated
                feature_map = torch.cat((feature_map, data.clone()), dim=0)

    print(feature_map.shape)        # just so we can see what it looks like 

    # here, the feature maps will have already been collected and thus we send them to become PCA 
    from utilsPCA import generate_and_set_pca_broadcast_equal_int
    generate_and_set_pca_broadcast_equal_int(
        current_feature=target_conv,
        current_fmap=feature_map,
        logger=logger
    )

print(f'PCA int completed')

# endregion



# region train 
from utils_v2 import train
import torch.optim as optim

# creating the argDict for the training 
argDict = {
            'lr': LR,
            'maxEpoch': MAX_EPOCH,
            'idleEpoch': IDLE_EPOCH,
            'outputName': MODEL_NAME,
            'optimizer': optim.SGD(model.parameters(), lr=LR),
            'criterion': nn.CrossEntropyLoss(),
            'logger': logger
        }

print(f'training starts here')


from datetime import datetime
start_time = datetime.now()

outputDict = train(
    model=model,
    argDict=argDict,
    givenDataloader=train_dataloader,
    evalDataloader=val_dataloader,
    testDataloader=test_dataloader
)

end_time = datetime.now()
time_difference = end_time - start_time
print(f'the time taken to train is {time_difference}')

from utils_v2 import load_model_from_file, test, save_dict_to_file
# loading the best version, and then testing it 
model = load_model_from_file(model, argDict['outputName'], argDict['outputName'])
test_accuracy = test(
    model=model,
    argDict=argDict,
    givenDataloader=test_dataloader
)
tempString = 'testing accuracy of ' + argDict['outputName'] + " is: " + str(test_accuracy)
print(tempString)

argDict['test_accuracy'] = str(test_accuracy)
save_dict_to_file(outputDict, argDict['outputName'], argDict['outputName'])

# endregion