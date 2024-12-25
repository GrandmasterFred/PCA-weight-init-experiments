# this file will load and train a basic resnet18 implementation. This will be where we test out the vram size and how much we can shove on it while maintaining good training stuff 
# to make it faster, we will be referencing D:\gitprojects\PCA_paper_exp_data\load_to_gpu_DEV.py

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
LR = 0.0008         # slightly lowered due to bumped up batch size 
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
# transform = transforms.Compose([
#             transforms.Resize((224, 224)),  # InceptionV3 input size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for single channel
#         ])

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to InceptionV3 input size
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize for single grayscale channel
])

from utils_PCA_paper_exp_data import get_memory_info
memory_info = get_memory_info()
print(f'before loading dataset {memory_info}')

from utils_PCA_paper_exp_data import custom_dataset_compcars_loadtoVRAM
train_dataset, val_dataset, test_dataset = custom_dataset_compcars_loadtoVRAM(
    root_dir = DATA_LOC, 
    transform = transform, 
    seed=RANDOM_SEED,
    memcheck=True   # this is the debug thingy where i make it print out the vram available
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
# Load the ResNet18 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

# Modify the first convolutional layer to accept 1 input channel
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

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

conv_layers_with_required_fmaps = [
    # first conv
    (model.conv1,               None,                   'conv1'),

    # first layer, 2 blocks inside 
    (model.layer1[0].conv1,     model.maxpool,          'lyr1_b1_conv1'),
    (model.layer1[0].conv2,     model.layer1[0].relu,     'lyr1_b1_conv2'),

    (model.layer1[1].conv1,     model.layer1[0].bn2,          'lyr1_b2_conv1'),
    (model.layer1[1].conv2,     model.layer1[1].relu,     'lyr1_b2_conv2'),

    # second layer, (block, downsample) -> block 
    (model.layer2[0].conv1,     model.layer1[1].bn2,          'lyr2_b1_conv1'),
    (model.layer2[0].conv2,     model.layer2[0].relu,     'lyr2_b1_conv2'),

    (model.layer2[0].downsample[0],     model.layer2[0].bn2,     'lyr2_ds_conv1'),

    (model.layer2[1].conv1,     model.layer2[0].downsample[1],          'lyr2_b2_conv1'),
    (model.layer2[1].conv2,     model.layer2[1].relu,     'lyr1_b2_conv2'),

    # # third layer, (block, downsample) -> block 
    # (model.layer3[0].conv1,     model.layer2[1].bn2,          'lyr3_b1_conv1'),
    # (model.layer3[0].conv2,     model.layer3[0].relu,     'lyr3_b1_conv2'),

    # (model.layer3[0].downsample[0],     model.layer3[0].bn2,     'lyr3_ds_conv1'),

    # (model.layer3[1].conv1,     model.layer3[0].downsample[1],          'lyr3_b2_conv1'),
    # (model.layer3[1].conv2,     model.layer3[1].relu,     'lyr3_b2_conv2'),

    # # fourth layer, (block, downsample) -> block 
    # (model.layer4[0].conv1,     model.layer3[1].bn2,          'lyr4_b1_conv1'),
    # (model.layer4[0].conv2,     model.layer4[0].relu,     'lyr4_b1_conv2'),

    # (model.layer4[0].downsample[0],     model.layer4[0].bn2,     'lyr4_ds_conv1'),

    # (model.layer4[1].conv1,     model.layer4[0].downsample[1],          'lyr4_b2_conv1'),
    # (model.layer4[1].conv2,     model.layer4[1].relu,     'lyr4_b2_conv2'),
]

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