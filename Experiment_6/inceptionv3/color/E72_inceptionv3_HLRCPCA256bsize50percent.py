# this file will load and train a basic resnet18 implementation. This will be where we test out the vram size and how much we can shove on it while maintaining good training stuff 

import torch
import random
import os
import torchvision.transforms as transforms
from utils_PCA_paper_exp_data import MyLogger_modified_for_PCA, create_sequential_folder
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

#region var
RANDOM_SEED= random.randint(1000, 9999)
DATA_LOC = 'data/compCarsSvData_enhanced'
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
LR = 0.0024         # slightly lowered due to bumped up batch size 
MAX_EPOCH = 150
IDLE_EPOCH = 10
OUTPUT_NAME = MODEL_NAME
# the optimizer and the criterion is at the argdict


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

from utils_PCA_paper_exp_data import custom_dataset_compcars_loadtoVRAM
train_dataset, val_dataset, test_dataset = custom_dataset_compcars_loadtoVRAM(
    root_dir = DATA_LOC, 
    transform = transform, 
    seed=RANDOM_SEED,
    memcheck=True   # this is the debug thingy where i make it print out the vram available
    )

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
model.to(device)

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
conv_layers_with_required_fmaps = return_percentage_of_list(conv_layers_with_required_fmaps, 50) 
# conv_layers_with_required_fmaps = conv_layers_with_required_fmaps[:1]

activation = {} # stores in dict 
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

from utils_dataset import create_pca_dataloader
pca_sample_loader = create_pca_dataloader(
    train_list=train_dataset.image_list_only_loc,
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
    try:
        from utilsPCA import generate_and_set_pca_broadcast_equal_int
        generate_and_set_pca_broadcast_equal_int(
            current_feature=target_conv,
            current_fmap=feature_map,
            logger=logger
        )
    except Exception as e:
        print(f'error when setting {str(target_conv)} for {target_name}, so will be skipping it')

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