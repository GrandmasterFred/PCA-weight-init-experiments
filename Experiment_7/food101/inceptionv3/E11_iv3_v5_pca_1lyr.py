# this file is the base training file for a working variation for the compcars training. We will test out if this works on the food, and aircraft dataset or not 

import torch
import random
import os
import torchvision.transforms as transforms
from utils_PCA_paper_exp_data import MyLogger_modified_for_PCA, create_sequential_folder
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from sklearn.model_selection import train_test_split

#region var
RANDOM_SEED= random.randint(1000, 9999)
DATA_LOC = '/ibm/gpfs/home/kpha0008/data/E11_fooddataset/food-101/food-101/'
DATA_NAMES = 'full'
TRAIN_FROM_SCRATCH = True
DETECT_ANON = False
MODEL_NAME = create_sequential_folder()    # this would just be the file name minus the extension
# everything would be saved based on this model name 

PCA_SAMPLES_PER_CLASS = 2
NUM_CLASSES = 281
IMPLEMENT_PCA = True
PCA_LAYERS = 4      # this is currently useless

BATCH_SIZE = 256
CHECK_DIST = False   # this simply checks if all the classes exists for the val and test set 

# for this, it takes prio in order of pca -> trans -> random int. Meaning if random int is needed, both pca and trans has to be false 
INITIALIZE_PCA = True
INITIALZE_TRANS_WEIGHTS = False

# this is for the arg dict
LR = 0.002         # slightly lowered due to bumped up batch size 
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
#region food dataset 

# this will be a cleaned up version that loads the food dataset directly into ram, and hopefully it works 

class e11_food_load_to_ram(Dataset):
    def __init__(self, image_list, class_path, transform):
        """
        Args:
            image_list (list): List of tuples (image_path, label)
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.

            this function loads all the image to vram, and keeps giving information on how much vram is still available
        """
        self.class_path = class_path            # text file that provides a list of manufac
        self.image_list = image_list
        self.transform = transform
        self.image_list_only_loc = [loc for loc, _ in image_list]   # this is needed for an external func
        self.images_transformed = False

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # setups self.label_dict that holds classname (folder name) -> label (label for nn training)
        self.label_dict = self.get_food_label_dict(self.class_path) 

        # creation of self.image_loc_family_list, encodes image_loc, class_name
        self.image_loc_family_list = []
        for img_loc, class_name in self.image_list:
            # first is loc, se
            self.image_loc_family_list.append((img_loc, self.label_dict[class_name]))
        
        self.image_loc_label = []
        # creating of image_loc_labe, this is to be used in the PCA function 
        for image_loc, manufac_name in self.image_list:
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

    def get_food_label_dict(self, file_path):
        '''
        this function reads the classes.txt provided in the food dataset, and then use that to generate a dict that encodes class_name => label
        '''
        label_dict = {}
        
        # Open and read the file line by line
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                # Strip any whitespace characters and add to dictionary with label starting from 0
                name = line.strip()
                label_dict[name] = idx
        
        return label_dict
        pass
    
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
        _, label = self.image_loc_family_list[idx]

        # get the image, which is already saved self.image_tensors
        image = self.list_of_pil_images[idx]    # this should be in PIL

        # transform the pil image
        tensor = self.transform(image)
        
        # convert the manufac name into label using dict. The dict label starts from 0 already, so no need to -1
        # label = self.label_dict[manufac]  # i dont need this anymore since we are directly getting the labels from self.image_loc_family_list

        return tensor, label     # where subfolder starts from 0

def e11_food_dataset(root_dir, transform, seed, class_path):
    '''
    this function accepts:
        root dir, transform, and memory check
    this function returns:
        tran_dataset, val_dataset, and test_dataset 
    '''

    # splits the dataset into train_set, val_set, and test_set, this process already splits it perfectly, so i dont have to validate it 
    train_set, val_set, test_set = split_data(root_dir, seed)

    # create the dataset, and load it to vram. This dataset only stores RAW images, and has not have them transformed yet. Raw images will be in tensor foramt 
    train_dataset = e11_food_load_to_ram(train_set, class_path,transform)
    val_dataset = e11_food_load_to_ram(val_set, class_path,transform)
    test_dataset = e11_food_load_to_ram(test_set, class_path,transform)

    # returns the datasets
    return train_dataset, val_dataset, test_dataset



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
            transforms.Resize((299, 299)),  # InceptionV3 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for single channel
        ])

# transform = transforms.Compose([
#     transforms.Resize((299, 299)),  # Resize to InceptionV3 input size
#     transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize for single grayscale channel
# ])

from utils_PCA_paper_exp_data import get_memory_info
memory_info = get_memory_info()
print(f'before loading dataset {memory_info}')

food_root = DATA_LOC
food_images = os.path.join(food_root, 'images')
food_classes = os.path.join(food_root, os.path.join('meta', 'classes.txt'))

train_dataset, val_dataset, test_dataset = e11_food_dataset(
    root_dir = food_images, 
    transform = transform, 
    seed=RANDOM_SEED,
    class_path = food_classes
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
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)

# # Modify the first convolutional layer to accept single-channel input
# model.Conv2d_1a_3x3.conv = nn.Conv2d(
#     in_channels=1,  # Change to 1 channel for grayscale
#     out_channels=model.Conv2d_1a_3x3.conv.out_channels,
#     kernel_size=model.Conv2d_1a_3x3.conv.kernel_size,
#     stride=model.Conv2d_1a_3x3.conv.stride,
#     padding=model.Conv2d_1a_3x3.conv.padding,
#     bias=model.Conv2d_1a_3x3.conv.bias is not None
# )

model.eval()
model.aux_logits = False

# endregion 



# region PCA int 
import torch
import torch.nn as nn
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux


def get_conv_layers_with_prev_inception(model):
    conv_layers = []
    previous_layer = None  # Track the last valid layer
    prev_basic_conv = None
    prev_inceptionA = None
    prev_inceptionB = None
    prev_inceptionC = None
    prev_inceptionD = None
    prev_inceptionE = None
    prev_inceptionAux = None
    instance_counter = 0    # Track whether we're at the first instance of a block
    first_instance = None   # this one is to track the first instance of a module 
    first_instance_name = ""
    
    # this is just a marker 
    current_basic_conv = None
    current_IA = None
    current_IB = None
    current_IC = None
    current_ID = None
    current_IE = None

    # Traverse all named modules
    for name, layer in model.named_modules():
        # If it's a Conv2d layer, store it along with the correct previous layer
        # TODO also check for if the name has conv2d or not 
        if isinstance(layer, nn.Conv2d):
            #print(f'{name}')

            if instance_counter == 0:
                conv_layers.append((layer, None, name))
            elif "Conv2d" in name:
                conv_layers.append((layer, prev_basic_conv, name))
            elif 'Mixed_' in name:
                # checks for if first layer
                if 'pool' in name:
                    print(f'special case, this should be half ish correct {name}')
                    try:
                        previous_layer = conv_block_for_first_mixed_layer
                    except Exception as e:
                        print(f'e')
                elif '_1' in name or name.count('_') == 1:
                    # checks for if needs conv or needs block 
                    if first_instance is None or first_instance_name == name.split('.')[0]:
                        # this means it is first instance, so it will always inherit conv for the first stage 
                        # this specific loop is to ensure that we do not take other conv layers, and we only keep the correct one we need 
                        if first_instance is None:
                            # this one locks in the previous basic conv 
                            previous_layer = prev_basic_conv
                            conv_block_for_first_mixed_layer = prev_basic_conv
                        else:
                            # this means that we dont have to keep changing the previous layer already
                            previous_layer = conv_block_for_first_mixed_layer

                        first_instance = True
                        first_instance_name = name.split('.')[0]

                    else:
                        # this means that this is not the first instance, so it will always inherit blocks 
                        # special conditions have to be coded in for clarity sake. I should have just used a single inception tracker instead of multiple i guess, my bad 
                        if any(x in name for x in ['5c', '5d', '5c', '5d']):
                            previous_layer = prev_inceptionA
                        elif '6a' in name:
                            previous_layer = current_IA
                        elif any(x in name for x in ['6b']):
                            previous_layer = current_IB
                        elif any(x in name for x in ['6c', '6d', '6e']):
                            previous_layer = prev_inceptionC
                        elif '7a' in name:
                            previous_layer = current_IC
                        elif any(x in name for x in ['7b']):
                            previous_layer = current_ID
                        elif any(x in name for x in ['7c']):
                            previous_layer = current_IE
                        elif 'InceptionAux' in name:
                            previous_layer = prev_inceptionAux
                else:
                    # this means that it is not the first layer anymore, so i can just take the most recent conv block or whatever 
                    previous_layer = prev_basic_conv
                # then appending it so that it works out correctly 
                conv_layers.append((layer, previous_layer, name))


            
            instance_counter += 1
            
        

        # Store the previous BasicConv2D or Inception layers separately
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, BasicConv2d)):
            prev_basic_conv = current_basic_conv        # so this sould be correctly offsetted now 
            current_basic_conv = layer
            previous_layer = layer
        elif isinstance(layer, InceptionA):
            prev_inceptionA = current_IA        # these must be offsetted as well 
            current_IA = layer
            previous_layer = layer
        elif isinstance(layer, InceptionB):
            prev_inceptionB = current_IB        # these must be offsetted as well 
            current_IB = layer
            previous_layer = layer
        elif isinstance(layer, InceptionC):
            prev_inceptionC = current_IC        # these must be offsetted as well 
            current_IC = layer
            previous_layer = layer
        elif isinstance(layer, InceptionD):
            prev_inceptionD = current_ID        # these must be offsetted as well 
            current_ID = layer
            previous_layer = layer
        elif isinstance(layer, InceptionE):
            prev_inceptionE = current_IE        # these must be offsetted as well 
            current_IE = layer
            previous_layer = layer
        elif isinstance(layer, InceptionAux):
            prev_inceptionAux = layer
            previous_layer = layer

        # Logic for handling pooling layers
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):     
            previous_layer = layer
        # If it's a valid, single layer, update the previous layer to this one
        elif not isinstance(layer, (nn.Sequential, nn.ModuleList, nn.ModuleDict, BasicConv2d)):
            previous_layer = layer

    # this is just to catch a bug, there could have been a more elegant way to make this, but I am too lazy
    row_as_list = list(conv_layers[0])
    row_as_list[1] = None
    conv_layers[0] = tuple(row_as_list)
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


# Extract convolution layers with their previous layers
conv_layers_with_required_fmaps = get_conv_layers_with_prev_inception(model)
# conv_layers_with_required_fmaps = return_percentage_of_list(conv_layers_with_required_fmaps, 50) 
conv_layers_with_required_fmaps = conv_layers_with_required_fmaps[:1]

activation = {} # stores in dict 
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

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

outputDict = train(
    model=model,
    argDict=argDict,
    givenDataloader=train_dataloader,
    evalDataloader=val_dataloader,
    testDataloader=test_dataloader
)

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

# this should be able to automatically plot the image and save it neatly
from utils_v2 import plot_and_save_graph
plot_and_save_graph(MODEL_NAME)