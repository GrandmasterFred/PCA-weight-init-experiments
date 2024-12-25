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
PCA_LAYERS = 4

BATCH_SIZE = 32
CHECK_DIST = False   # this simply checks if all the classes exists for the val and test set 

# for this, it takes prio in order of pca -> trans -> random int. Meaning if random int is needed, both pca and trans has to be false 
INITIALIZE_PCA = True
INITIALZE_TRANS_WEIGHTS = False

# this is for the arg dict
LR = 0.001
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
            transforms.Resize((100, 100)),  # InceptionV3 input size
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.485,), (0.229,))  # Normalize for single channel
        ])

# from utils_dataset import create_dataloader_compcars
# train_loader, val_loader, test_loader = create_dataloader_compcars(
#     data_loc = DATA_LOC, 
#     transform = transform, 
#     batch_size = BATCH_SIZE, 
#     gen=gen, 
#     split_ratio=0.8,
#     check_dist=CHECK_DIST)

# from dev_files.custom_dataloader_dev import custom_dataset_compcars

from utils_PCA_paper_exp_data import custom_dataset_compcars
train_dataset, val_dataset, test_dataset = custom_dataset_compcars(
    root_dir = DATA_LOC, 
    transform = transform, 
    seed=RANDOM_SEED)

# create dataloaders from those datasets 
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#endregion

#region network int
# no function needed for this, since this is a simple enough procedure 
from utils_network import originalPCAPaper
model = originalPCAPaper(num_classes=NUM_CLASSES)           # this will need to be changed based the dataset I guess
model.to(device=device)                                     # yeeting it to CUDA if available 

#endregion

#region train
'''
main things to collect are
    accuracy and loss values for train, val, and test 
    how many epochs does it take for training to complete 
    basically i can just use the previous training loops to make sure i have everything 

Given that i have the dataloaders separated,
    I can just have a massive loop that goes through the epochs, and then
    train -> val (repeat), and collect their data 
        Have to save the model when it is better compared to last iteration as well 
    and then finally a test loop
'''
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

#TODO an option to turn on weight freezing for the conv layers of the NN

# training the model, just using a previous function from utils
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


#endregion