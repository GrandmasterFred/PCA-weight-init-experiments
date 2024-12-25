import torch
import random
import os
import torchvision.transforms as transforms
from utils_PCA_paper_exp_data import MyLogger_modified_for_PCA, create_sequential_folder
import torchvision
from torch.utils.data import DataLoader

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
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#endregion

#region network int
# no function needed for this, since this is a simple enough procedure 
from utils_network import originalPCAPaper
model = originalPCAPaper(num_classes=NUM_CLASSES)           # this will need to be changed based the dataset I guess
model.to(device=device)                                     # yeeting it to CUDA if available 

#endregion

#region weight int
# this section should have switches and selectors to determine which is used pca, random int, or trans learn, if avaialble. I would probably just make three diff conditions that clearly outline which will be used 
if INITIALIZE_PCA:
    print(f'pca implementation')

    # the pca function goes here
elif INITIALZE_TRANS_WEIGHTS:
    print(f'transfer learning implementation')
else:
    print(f'random weight implementation')

from utils_dataset import create_pca_dataloader

# using a smaller dataset temporarily
pca_sample_loader = create_pca_dataloader(
    train_list=train_dataset.image_list_only_loc,
    samples_per_label=PCA_SAMPLES_PER_CLASS,
    transform=transform,
    num_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE
)


# this part now starts using dataloader, as well as the model to start pca initialization
from tqdm import tqdm

feature_map = None

# gathering the first set of feature maps for the pca 
for idx, (data, label) in enumerate(tqdm(pca_sample_loader)):
    data = data.to(device)

    if feature_map is None:
        # this one just assumes the other feature maps are none as well
        feature_map = data.clone()
    else:
        # meaning the map is not empty, and as such can be concatenated
        feature_map = torch.cat((feature_map, data.clone()), dim=0)

from utils_PCA import check_memory_usage_tensor
check_memory_usage_tensor(feature_map)
'''
==============================================
targeted conv layers  
==============================================
'''
# each item in this list is a tuple that encodes the target conv layer, and the features that comes before it. 
conv_layers_with_required_fmaps = [
    (model.features1[0], None),
    (model.features2[0], model.features1),
    (model.features3[0], model.features2),
]

'''
==============================================
first layer 
==============================================
'''
print(f'first layer')

# targeting the conv block, which is the first of the features 
current_feature = conv_layers_with_required_fmaps[0][0]
current_fmap = feature_map      # this could have been [0][1] but since it is none, it should be targeted at the original feature map, which is just the sample PCA images 

# recording the model's state dict to see if anything changed or not 
save1 = str(model.state_dict())

from utilsPCA import generate_and_set_pca_broadcast_equal_int
generate_and_set_pca_broadcast_equal_int(
    current_feature=current_feature,
    current_fmap=current_fmap,
    logger=logger
)

save2 = str(model.state_dict())

if save1 == save2:
    print(f'the save states are the same, the process failed')
else:
    print(f'save states are different, it worked')

'''
==============================================
second layer  
==============================================
'''
# instead of the feature map, this layer would instead collect the output feature maps that has been modified, in a way i guess 
print(f'SECOND LAYER')
import torch.nn as nn
class fmap_generator_class(nn.Module):
    def __init__(self, list_of_convs) -> None:
        super(fmap_generator_class, self).__init__()
        # Create a sequential block from the list of layers
        self.layer_block = nn.Sequential(*list_of_convs)
    
    def forward(self, x):
        # Forward pass through all the layers in the sequential block
        return self.layer_block(x)

# this should have the weights that are modified by the PCA already
fmap_generator_second_layer = fmap_generator_class(
    list_of_convs=[
        model.features1
    ]
    )

# collects all the feature maps 
feature_map = None

# gathering the first set of feature maps for the pca 
for idx, (data, label) in enumerate(tqdm(pca_sample_loader)):
    data = data.to(device)

    if feature_map is None:
        # this one just assumes the other feature maps are none as well
        feature_map = fmap_generator_second_layer(data).detach()
    else:
        # meaning the map is not empty, and as such can be concatenated
        feature_map = torch.cat((feature_map, fmap_generator_second_layer(data).detach()), dim=0)

check_memory_usage_tensor(feature_map)
print(f'layer two fmap shape {feature_map.shape}')

# targeting the conv block, which is the first of the features 
current_feature = conv_layers_with_required_fmaps[1][0]
current_fmap = feature_map      # this could have been [0][1] but since it is none, it should be targeted at the original feature map, which is just the sample PCA images 

# recording the model's state dict to see if anything changed or not 
save1 = str(model.state_dict())

from utilsPCA import generate_and_set_pca_broadcast_equal_int
generate_and_set_pca_broadcast_equal_int(
    current_feature=current_feature,
    current_fmap=current_fmap,
    logger=logger
)

save2 = str(model.state_dict())

if save1 == save2:
    print(f'the save states are the same, the process failed')
else:
    print(f'save states are different, it worked')


'''
==============================================
third layer  
==============================================
'''
print(f'third LAYER')

# this should have the weights that are modified by the PCA already
# this list of convs could be incrementally added onto for each layer to automate it 
fmap_generator_third_layer = fmap_generator_class(
    list_of_convs=[
        model.features1,
        model.features2
    ]
    )

# collects all the feature maps 
feature_map = None

# gathering the first set of feature maps for the pca 
for idx, (data, label) in enumerate(tqdm(pca_sample_loader)):
    data = data.to(device)

    if feature_map is None:
        # this one just assumes the other feature maps are none as well
        feature_map = fmap_generator_third_layer(data).detach()
    else:
        # meaning the map is not empty, and as such can be concatenated
        feature_map = torch.cat((feature_map, fmap_generator_third_layer(data).detach()), dim=0)

check_memory_usage_tensor(feature_map)
print(f'layer three fmap shape {feature_map.shape}')

# targeting the conv block, which is the first of the features 
current_feature = conv_layers_with_required_fmaps[2][0]
current_fmap = feature_map      # this could have been [0][1] but since it is none, it should be targeted at the original feature map, which is just the sample PCA images 

# recording the model's state dict to see if anything changed or not 
save1 = str(model.state_dict())

from utilsPCA import generate_and_set_pca_broadcast_equal_int
generate_and_set_pca_broadcast_equal_int(
    current_feature=current_feature,
    current_fmap=current_fmap,
    logger=logger
)

save2 = str(model.state_dict())

if save1 == save2:
    print(f'the save states are the same, the process failed')
else:
    print(f'save states are different, it worked')


#endregion

#region freeze weights
# Freeze features1
for param in model.features1.parameters():
    param.requires_grad = False

# Freeze features2
for param in model.features2.parameters():
    param.requires_grad = False

# Freeze features3
for param in model.features3.parameters():
    param.requires_grad = False


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
            'optimizer': optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR),
            'criterion': nn.CrossEntropyLoss(),
            'logger': logger
        }

print(f'training starts here')



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