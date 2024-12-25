# will pass the arguments into here as a dictionary
# make a class that has the logger, as well as the method of printing i guess
# this is the updated utils file 19 10 2023
'''
changes:
added model.train() to training
added model.eval() to eval and testing

added function that saves model parameters (like optimizers and the like) when saving the models
added function that allows the loading of those parameters
'''
import torch
import numpy as np
import logging


import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import os
from torchvision.io import read_image
import random
from scipy.io import loadmat

class fetch_name_and_img:
    def __init__(self, mat_location, folder_location):
        # mat location refers to the file that encodes folders to name
        # folder location refers to the root of the file
        self.mat_location = mat_location
        self.folder_location = folder_location

        # i would like to load up the matrix location, and then use that first
        try:
            # Load the .mat file
            self.mat_data = loadmat(self.mat_location)
            self.mat_data = self.mat_data['sv_make_model_name']

        except Exception as e:
            print(f"Error loading MAT file: {e}")

    def make_model_combine(self, class_number):
        # this one just combines the first two column, and returns the car label
        # the class number will be from 1-281, whereas the labelling goes from 0-280
        class_number = int(class_number) - 1

        label = str(self.mat_data[class_number, 0]) + " " +str(self.mat_data[class_number, 1])
        return label

    def plot(self, imgs, row_title=None, **imshow_kwargs):
        # adapted from, https://github.com/pytorch/vision/tree/main/gallery/
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                boxes = None
                masks = None
                if isinstance(img, tuple):
                    img, target = img
                    if isinstance(target, dict):
                        boxes = target.get("boxes")
                        masks = target.get("masks")
                    elif isinstance(target, tv_tensors.BoundingBoxes):
                        boxes = target
                    else:
                        raise ValueError(f"Unexpected target type: {type(target)}")
                img = F.to_image(img)
                if img.dtype.is_floating_point and img.min() < 0:
                    # Poor man's re-normalization for the colors to be OK-ish. This
                    # is useful for images coming out of Normalize()
                    img -= img.min()
                    img /= img.max()

                img = F.to_dtype(img, torch.uint8, scale=True)
                if boxes is not None:
                    img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
                if masks is not None:
                    img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

                ax = axs[row_idx, col_idx]
                ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()

    def select_random_image(self, folder_path):
        if not os.path.isdir(folder_path):
            print(f"{folder_path} is not a valid directory.")
            return None

        # Get a list of all files in the directory
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Filter files to include only images (you can adjust the list of valid image extensions)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]

        # Check if there are any images in the folder
        if not image_files:
            print(f"No image files found in {folder_path}.")
            return None

        # Select a random image from the list
        random_image = random.choice(image_files)

        # Return the path to the selected random image
        return os.path.join(folder_path, random_image)

    def fetch_img(self, folder_names):
        # this function takes in a list of folder names, prints the labels and images, and returns a tuples of it i guess

        # converts the folder location to string, so that i can use it later in paths
        for i in range(len(folder_names)):
            if isinstance(folder_names[i], int):
                try:
                    folder_names[i] = str(folder_names[i])
                except ValueError:
                    (f"Unable to convert '{folder_names[i]}' to an integer.")

        # getting the corresponding labels
        # i need to stick the first and second column together to get make + model
        labels = []
        for i in folder_names:
            labels.append(self.make_model_combine(i))
        print(labels)

        # fetches random image for each folder name
        img = []
        for idx, name in enumerate(folder_names):
            # this will use torchvision.io read_image, since it is quite useful
            img_location = os.path.join(self.folder_location, folder_names[idx])
            img_location = self.select_random_image(img_location)
            print(img_location)
            img.append(read_image(img_location))

        # plotting the image
        self.plot(img)


def load_model_from_file(model, folder_path, filename=None):
    import os
    import torch
    # Add the ".pth" extension to the filename if missing
    # if not filename.endswith(".pth"):
    #     filename += ".pth"

    if filename is None:
        # this means that the folder path is already the file name
        save_path = folder_path
    else:
        # getting the filename of the path
        if not filename.endswith(".pth"):
            filename += ".pth"

        save_path = os.path.join(folder_path, filename)

    # loading it up
    try:
        checkpoint = torch.load(save_path)
        if 'model_state_dict' in checkpoint:
            # this one should just check if it is true or not
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    except FileNotFoundError:
        print('file is not found')
    except Exception as e:
        print('a weird error: ', e)

    return model

def save_model_to_file(model, folder_path, filename, optimizer, epoch):
    import torch
    import os
    # Check if the folder exists, and create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Add the ".pth" extension to the filename if missing
    if not filename.endswith(".pth"):
        filename += ".pth"

    # Save the model to the specified path
    save_path = os.path.join(folder_path, filename)
    # this section was changed so that it saves all the parameters now as well
    #torch.save(model.state_dict(), save_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def save_dict_to_file(dict, folder_path, filename):
    import json
    import os
    def is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except:
            return False

    def copy_dict_with_serializable_items(original_dict):
        new_dict = {}
        for key, value in original_dict.items():
            if is_json_serializable(value):
                new_dict[key] = value
        return new_dict

    # Check if the folder exists, and create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Add the ".pth" extension to the filename if missing
    if not filename.endswith(".json"):
        filename += ".json"

    # modify the .json file so that it can be saved to file
    new_dict = copy_dict_with_serializable_items(dict)

    # Save the dictionary to a JSON file
    filename = os.path.join(folder_path, filename)
    with open(filename, "w") as json_file:
        json.dump(new_dict, json_file)

class MyLogger:
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
        # this then also prints to console as well
        print(message)


# this is the evaluation fucntion, where it trains it for an epoch, and returns the validation accuracy
def eval(model, argDict, givenDataloader):
    import numpy as np
    # setting the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # forces it to train on cpu real quick
    # device = torch.device('cpu')
    model.to(device)
    model.eval()

    eval_accuracy = 0
    eval_loss = 0

    # setting evaluation mode
    with torch.no_grad():
        accuracy_values = []
        loss_values = []
        for idx, (data, label) in enumerate(givenDataloader):
            try:
                data = data.to(device)
                label = label.to(device)

                # getting the predictions
                outputs = model(data)

                # getting the loss as well
                loss = argDict['criterion'](outputs, label)
                loss_values.append((loss.item()))

                # getting the accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label).float().mean()
                accuracy_values.append(accuracy)
            except Exception as e:
                # this section logs the whatever
                errString = f'error located at index: {str(idx)} at epoch evalLoop'
                argDict['logger'].log(str(errString))
                argDict['logger'].log(str(e))

        # calculating the accuracy
        eval_loss = np.mean(loss_values)
        eval_accuracy = torch.mean(torch.stack(accuracy_values))

    return eval_accuracy, eval_loss

def test(model, argDict, givenDataloader):
    # this is basically the same as the evaluation one, but it is just given a different name to make things easier i guess. There should be no reason that they are two separate functions
    # setting the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()

    test_accuracy = 0

    # setting evaluation mode
    with torch.no_grad():
        accuracy_values = []
        for idx, (data, label) in enumerate(givenDataloader):
            try:
                data = data.to(device)
                label = label.to(device)

                # getting the predictions
                outputs = model(data)

                # getting the accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label).float().mean()
                accuracy_values.append(accuracy)
            except Exception as e:
                # this section logs the whatever
                errString = f'error located at index: {str(idx)} at testing section'
                argDict['logger'].log(str(errString))
                argDict['logger'].log(str(e))


        # calculating the accuracy
        test_accuracy = torch.mean(torch.stack(accuracy_values))

    return test_accuracy

def train(model, argDict, givenDataloader, evalDataloader=None, testDataloader=None):
    # section to check for presence of all the needed loaders
    import time
    start_time = time.time()

    if evalDataloader is None:
        print('you forgot eval loader')
        return
    if testDataloader is None:
        print('you forgot test laoder')
        return

    # get all the stuff out
    # update the learning rate of the optimizer
    for param_group in argDict['optimizer'].param_groups:
        param_group['lr'] = argDict['lr']

    # get the device type, and set it to cuda
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # casting model to device
    model.to(device)
    model.train()

    # training for multiple epochs
    epoch_accuracy_values_train = []
    epoch_accuracy_values_eval = []
    epoch_loss_values_train = []
    epoch_loss_values_eval = []


    best_epoch_value = 0
    best_epoch_epoch = 0

    for currentEpoch in range(argDict['maxEpoch']):
        accuracy_values = []
        loss_values = []

        has_nan = False

        for idx, (data, label) in enumerate(givenDataloader):
            #  this is captured inside a try because of some weird thing breaking in the middle sometimes when there is only 1 label
            try:
                data, label = data.to(device), label.to(device)

                # this will be the training loop
                outputs = model(data)

                loss = argDict['criterion'](outputs, label)

                # backward pass and optimization
                argDict['optimizer'].zero_grad()
                loss.backward()

                # stabalize the loss, this is to prevent NaNs, 2023 12 28
                # https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/clip_grad.py
                # this is to prevent the NaN issues. 
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

                # only trigger this is the loss has any NaN values, because it seems to be slowing down the training significantly 
                # i have a feeling that this might be because of the high initialized values of the weights that are causing this
                # i could try out methods that makes sure that the standard diviation of the weights are the same? I guess
                try:
                    check_is_nan = torch.isnan(loss)
                    if True in check_is_nan:
                        # print('nan value detected')
                        # print(f'loss is {loss}, and the items {loss.item()}')
                        has_nan = True
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                except Exception as e:
                    print(e)
                    

                argDict['optimizer'].step()

                # data logging phase, obtains loss and accuracy
                loss_values.append((loss.item()))

                # getting the accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label).float().mean()
                accuracy_values.append(accuracy)
            except Exception as e:
                # this section logs the whatever
                errString = f'error located at index: {str(idx)} at epoch {str(currentEpoch)}'
                argDict['logger'].log(str(errString))
                argDict['logger'].log(str(e))

        # quick error message if it has nan 
        if has_nan:
            print(f'this epoch of {currentEpoch} resulted in a NaN at its loss function, and as used the function of clip_grad_norm_')

        # calculating epoch losses
        epoch_loss = np.mean(loss_values)
        epoch_loss_values_train.append(epoch_loss)
        epoch_accuracy = torch.mean(torch.stack(accuracy_values))   # due to it being tensor
        epoch_accuracy_values_train.append(epoch_accuracy.item())

        tempString = 'currently at epoch ' + str(currentEpoch) + ' train accuracy: ' + str(epoch_accuracy) + ' loss of: ' + str(epoch_loss)

        # this section is for evaluation of the model on the eval set
        eval_accuracy, eval_loss = eval(model, argDict, evalDataloader)
        epoch_accuracy_values_eval.append(eval_accuracy.item())
        epoch_loss_values_eval.append(eval_loss)

        # log it as well
        tempString = tempString + ' eval accuracy: ' + str(eval_accuracy)
        argDict['logger'].log(tempString)

        # if it improves, no need to break, else, break after reaching max idle epoch
        if eval_accuracy > best_epoch_value:
            best_epoch_value = eval_accuracy
            best_epoch_epoch = currentEpoch
            # save the model as well
            save_model_to_file(model, argDict['outputName'], argDict['outputName'], argDict['optimizer'], currentEpoch)
        else:
            if (currentEpoch - best_epoch_epoch) > argDict['idleEpoch']:
                # this means that this is the max trained  epoch
                break

    argDict['epoch_loss_values_train'] = epoch_loss_values_train
    argDict['epoch_loss_values_eval'] = epoch_loss_values_eval
    argDict['epoch_accuracy_values_train'] = epoch_accuracy_values_train
    argDict['epoch_accuracy_values_eval'] = epoch_accuracy_values_eval
    argDict['trainingStopEpoch'] = currentEpoch

    # records the time taken for all these
    end_time = time.time()
    elapsed_time = end_time - start_time
    argDict['elapsed_time'] = elapsed_time

    # saves the dictionary as well
    return argDict

def freq_save_train(model, argDict, givenDataloader, evalDataloader=None, testDataloader=None):
    # this function is a deviation of train, so that this is able to save every few epochs to show the progression of the model weights
    import time
    start_time = time.time()

    if evalDataloader is None:
        print('you forgot eval loader')
        return
    if testDataloader is None:
        print('you forgot test laoder')
        return

    # get all the stuff out
    # update the learning rate of the optimizer
    for param_group in argDict['optimizer'].param_groups:
        param_group['lr'] = argDict['lr']

    # get the device type, and set it to cuda
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # casting model to device
    model.to(device)
    model.train()

    # training for multiple epochs
    epoch_accuracy_values_train = []
    epoch_accuracy_values_eval = []
    epoch_loss_values_train = []
    epoch_loss_values_eval = []


    best_epoch_value = 0
    best_epoch_epoch = 0

    for currentEpoch in range(argDict['maxEpoch']):
        accuracy_values = []
        loss_values = []

        has_nan = False

        for idx, (data, label) in enumerate(givenDataloader):
            #  this is captured inside a try because of some weird thing breaking in the middle sometimes when there is only 1 label
            try:
                data, label = data.to(device), label.to(device)

                # this will be the training loop
                outputs = model(data)

                loss = argDict['criterion'](outputs, label)

                # backward pass and optimization
                argDict['optimizer'].zero_grad()
                loss.backward()

                # stabalize the loss, this is to prevent NaNs, 2023 12 28
                # https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/clip_grad.py
                # this is to prevent the NaN issues. 
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

                # only trigger this is the loss has any NaN values, because it seems to be slowing down the training significantly 
                # i have a feeling that this might be because of the high initialized values of the weights that are causing this
                # i could try out methods that makes sure that the standard diviation of the weights are the same? I guess
                try:
                    check_is_nan = torch.isnan(loss)
                    if True in check_is_nan:
                        # print('nan value detected')
                        # print(f'loss is {loss}, and the items {loss.item()}')
                        has_nan = True
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                except Exception as e:
                    print(e)
                    

                argDict['optimizer'].step()

                # data logging phase, obtains loss and accuracy
                loss_values.append((loss.item()))

                # getting the accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label).float().mean()
                accuracy_values.append(accuracy)
            except Exception as e:
                # this section logs the whatever
                errString = f'error located at index: {str(idx)} at epoch {str(currentEpoch)}'
                argDict['logger'].log(str(errString))
                argDict['logger'].log(str(e))

        # quick error message if it has nan 
        if has_nan:
            print(f'this epoch of {currentEpoch} resulted in a NaN at its loss function, and as used the function of clip_grad_norm_')

        # calculating epoch losses
        epoch_loss = np.mean(loss_values)
        epoch_loss_values_train.append(epoch_loss)
        epoch_accuracy = torch.mean(torch.stack(accuracy_values))   # due to it being tensor
        epoch_accuracy_values_train.append(epoch_accuracy.item())

        tempString = 'currently at epoch ' + str(currentEpoch) + ' train accuracy: ' + str(epoch_accuracy) + ' loss of: ' + str(epoch_loss)

        # this section is for evaluation of the model on the eval set
        eval_accuracy, eval_loss = eval(model, argDict, evalDataloader)
        epoch_accuracy_values_eval.append(eval_accuracy.item())
        epoch_loss_values_eval.append(eval_loss)

        # log it as well
        tempString = tempString + ' eval accuracy: ' + str(eval_accuracy)
        argDict['logger'].log(tempString)

        # if it improves, no need to break, else, break after reaching max idle epoch
        if eval_accuracy > best_epoch_value:
            best_epoch_value = eval_accuracy
            best_epoch_epoch = currentEpoch
            # save the model as well
            save_model_to_file(model, argDict['outputName'], argDict['outputName'], argDict['optimizer'], currentEpoch)
        else:
            if (currentEpoch - best_epoch_epoch) > argDict['idleEpoch']:
                # this means that this is the max trained  epoch
                break
        
        # saves when epoch+1 is divisible by 5
        try:
            if (currentEpoch + 1) % 5 == 0:
                temp_name = argDict.get('outputName', 'default_name')
                temp_name = f"epoch_{currentEpoch+1}_{str(temp_name)}"
                # save model to file -> model, folder, filename, optim, epoch
                save_model_to_file(model, argDict['outputName'], temp_name, argDict['optimizer'], currentEpoch)
        except Exception as e:
            print(f'issue with saving the model at 5 interval {e}')


    argDict['epoch_loss_values_train'] = epoch_loss_values_train
    argDict['epoch_loss_values_eval'] = epoch_loss_values_eval
    argDict['epoch_accuracy_values_train'] = epoch_accuracy_values_train
    argDict['epoch_accuracy_values_eval'] = epoch_accuracy_values_eval
    argDict['trainingStopEpoch'] = currentEpoch

    # records the time taken for all these
    end_time = time.time()
    elapsed_time = end_time - start_time
    argDict['elapsed_time'] = elapsed_time

    # saves the dictionary as well
    return argDict


def check_folder_exists(folder_name):
    import os

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        return
    return


import os
import json
import re
import matplotlib.pyplot as plt

def plot_and_save_graph(file_name):
    '''
    Function to plot and save a graph based on data in a JSON file.
    
    Args:
    file_name (str): The base name of the folder and file to target.

    Steps:
    1. Checks for the existence of a folder with the given name.
    2. Looks for a JSON file inside the folder with the same base name.
    3. Reads the JSON file to extract the required data.
    4. Plots train and eval accuracy values and a line for the extracted test accuracy.
    5. Saves the plot in the same folder as file_name.png.
    '''
    # Step 1: Check if folder exists
    if not os.path.isdir(file_name):
        print(f"Error: Folder '{file_name}' does not exist.")
        return

    # Step 2: Look for the JSON file
    json_path = os.path.join(file_name, f"{file_name}.json")
    if not os.path.isfile(json_path):
        print(f"Error: JSON file '{file_name}.json' not found in folder '{file_name}'.")
        return

    # Step 3: Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract test accuracy value
    test_accuracy = data.get("test_accuracy", "")
    match = re.search(r"tensor\(([\d\.]+)", test_accuracy)
    if match:
        extracted_test_acc = float(match.group(1))
    else:
        print("Error: Unable to extract test accuracy value.")
        return

    # Extract train and eval accuracy lists
    train_accuracies = data.get("epoch_accuracy_values_train", [])
    eval_accuracies = data.get("epoch_accuracy_values_eval", [])

    # Step 4: Plot train and eval accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label="Train Accuracy", marker='o', linestyle='-')
    plt.plot(eval_accuracies, label="Eval Accuracy", marker='x', linestyle='--')

    # Step 5: Plot extracted test accuracy as a horizontal line
    plt.axhline(y=extracted_test_acc, color='r', linestyle='-', label=f"Test Accuracy ({extracted_test_acc:.4f})")

    # Add labels, legend, and title
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Plot for {file_name}")
    plt.legend()
    plt.grid(True)

    # Step 6: Save the plot as a PNG file
    output_path = os.path.join(file_name, f"{file_name}.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid display issues

    print(f"Plot saved successfully as '{output_path}'.")