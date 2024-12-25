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
# transform = transforms.Compose([
#             transforms.Resize((299, 299)),  # InceptionV3 input size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for single channel
#         ])

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to InceptionV3 input size
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize for single grayscale channel
])

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

# Modify the first convolutional layer to accept single-channel input
model.Conv2d_1a_3x3.conv = nn.Conv2d(
    in_channels=1,  # Change to 1 channel for grayscale
    out_channels=model.Conv2d_1a_3x3.conv.out_channels,
    kernel_size=model.Conv2d_1a_3x3.conv.kernel_size,
    stride=model.Conv2d_1a_3x3.conv.stride,
    padding=model.Conv2d_1a_3x3.conv.padding,
    bias=model.Conv2d_1a_3x3.conv.bias is not None
)

model.eval()
model.aux_logits = False

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