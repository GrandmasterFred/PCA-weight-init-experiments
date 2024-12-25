import torch
import torch.nn as nn
from utils_network import originalPCAPaper  # Replace with your actual import
from utils_network import secondPCAPaper

def test_model(dummy_input, model):
    # Pass the dummy input through the model to ensure it works
    try:
        with torch.no_grad():  # No need to compute gradients for this check
            output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
    except Exception as e:
        print(f"Error during model forward pass: {e}")

    # 1. Count the total number of layers (including all types)
    total_layers = sum(1 for _ in model.modules())

    # 2. Count the number of convolutional layers
    conv_layers = sum(1 for layer in model.modules() if isinstance(layer, nn.Conv2d))

    # 3. Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # 4. Calculate the total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Output the results
    print(f'Total number of layers (including all types): {total_layers}')
    print(f'Number of convolutional layers: {conv_layers}')
    print(f'Total number of parameters: {total_params}')
    print(f'Total number of trainable parameters: {trainable_params}')

# Load the model
model = originalPCAPaper(num_classes=281)  # Adjust based on your model's requirements
model.eval()  # Set the model to evaluation mode
dummy_input = torch.randn(1, 1, 100, 100)  
test_model(dummy_input=dummy_input, model=model)


# model = secondPCAPaper(num_classes=281)
# model.eval()
# dummy_input = torch.randn(1, 1, 144, 144)  

#region custom small img net

from utils_network import custom_1conv
model = custom_1conv(num_classes=281, in_channels=1)  # Adjust based on your model's requirements
model.eval()  # Set the model to evaluation mode
dummy_input = torch.randn(1, 1, 100, 100)  
test_model(dummy_input=dummy_input, model=model)

from utils_network import custom_4conv
model = custom_4conv(num_classes=281, in_channels=1)  # Adjust based on your model's requirements
model.eval()  # Set the model to evaluation mode
dummy_input = torch.randn(1, 1, 100, 100)  
test_model(dummy_input=dummy_input, model=model)

from utils_network import custom_6conv
model = custom_6conv(num_classes=281, in_channels=1)  # Adjust based on your model's requirements
model.eval()  # Set the model to evaluation mode
dummy_input = torch.randn(1, 1, 100, 100)  
test_model(dummy_input=dummy_input, model=model)

from utils_network import custom_8conv
model = custom_8conv(num_classes=281, in_channels=1)  # Adjust based on your model's requirements
model.eval()  # Set the model to evaluation mode
dummy_input = torch.randn(1, 1, 100, 100)  
test_model(dummy_input=dummy_input, model=model)

#endregion

#region available networks 
from utils_network import alex_net_model, alex_net_transform, alex_net_transform_from_tensor
model = alex_net_model(pretrained=False)
model.eval()
dummy_input = torch.randn(1, 3, 100, 100)
transform = alex_net_transform_from_tensor()
dummy_input = transform(dummy_input)
test_model(dummy_input=dummy_input, model=model)

from utils_network import resnet_model, resnet_transform_from_tensor
print(f'all resnet versions')
resnet_names = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
]
for item in resnet_names:
    print(item)
    model = resnet_model(resnet_version=item,pretrained=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 100, 100)
    transform = resnet_transform_from_tensor()
    dummy_input = transform(dummy_input)
    test_model(dummy_input=dummy_input, model=model)



from utils_network import vgg_model, vgg_transform_from_tensor
print(f'all vgg versions')
vgg_names = [
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn'
]
for item in vgg_names:
    print(item)
    model = vgg_model(vgg_version=item,pretrained=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 100, 100)
    transform = vgg_transform_from_tensor()
    dummy_input = transform(dummy_input)
    test_model(dummy_input=dummy_input, model=model)

from utils_network import inceptionV3_model, inceptionV3_transform_from_tensor
model = inceptionV3_model(pretrained=False)
model.eval()
dummy_input = torch.randn(1, 3, 100, 100)
transform = inceptionV3_transform_from_tensor()
dummy_input = transform(dummy_input)
test_model(dummy_input=dummy_input, model=model)


from utils_network import inception_resnetv2_model, inception_resnetv2_transform_from_tensor
model = inception_resnetv2_model()      # no pre-training for this one 
model.eval()
dummy_input = torch.randn(1, 3, 100, 100)
transform = inception_resnetv2_transform_from_tensor()
dummy_input = transform(dummy_input)
test_model(dummy_input=dummy_input, model=model)

#endregion