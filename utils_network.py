# this file holds network declaractions, as well as network initializations. It should just return a nice and pure model 
import torch.nn as nn
import torch

#region pca paper networks

# first paper mimicry model, referenced from D:\gitprojects\PCAFeatureExtraction\PCAFeatureExtractionNewImplementation.ipynb using https://ieeexplore.ieee.org/abstract/document/8376025
class originalPCAPaper(nn.Module):
    # this is the original PCA paper https://ieeexplore.ieee.org/abstract/document/8376025
    # this is set to 38 since that is what the paper mentioned i guess
    # this one assumes input of size 1x100x100

    # this one is furthur modified that that it returns the feature maps for each of the conv layers i guess
    def __init__(self, num_classes: int = 38):
        super(originalPCAPaper, self).__init__()

        # self.convtest = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(7,7), stride=(1,1 ))

        # defining the layers of the neural network
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(7,7), stride=(1,1 )),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(10, 10), stride=(1,1 )),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=60, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(3840, num_classes)     # seems like i dont need a softmax function, since it automatically does it with cross entropy loss
        )

    def forward(self, x):
        # takes the output, then clones the tensor
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)

        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        # this was the prev method of getting all the needed fmaps, not needed anymore
        #return out, feature_map1, feature_map2, feature_map3
        return out



# this is added in for the grayscale vs color test 
class originalPCAPaper_RGB(nn.Module):
    # this is the original PCA paper https://ieeexplore.ieee.org/abstract/document/8376025
    # this is set to 38 since that is what the paper mentioned i guess
    # this one assumes input of size 1x100x100

    # this one is furthur modified that that it returns the feature maps for each of the conv layers i guess
    def __init__(self, num_classes: int = 38):
        super(originalPCAPaper_RGB, self).__init__()

        #self.convtest = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(7,7), stride=(1,1 ))

        # defining the layers of the neural network
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(7,7), stride=(1,1 )),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(10, 10), stride=(1,1 )),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=60, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(3840, num_classes)     # seems like i dont need a softmax function, since it automatically does it with cross entropy loss
        )

    def forward(self, x):
        # takes the output, then clones the tensor
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)

        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        # this was the prev method of getting all the needed fmaps, not needed anymore
        #return out, feature_map1, feature_map2, feature_map3
        return out


# second paper mimicry model 
# https://ieeexplore.ieee.org/document/9109741
class secondPCAPaper(nn.Module):
    # this is the original PCA paper https://ieeexplore.ieee.org/document/9109741
    # this is set to 38 since that is what the paper mentioned i guess
    # this one assumes input of size 1x100x100

    # this one is furthur modified that that it returns the feature maps for each of the conv layers i guess
    def __init__(self, num_classes: int = 38):
        super(secondPCAPaper, self).__init__()

        # defining the layers of the neural network
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,5), stride=(1,1 )),    #output of 10@140 140 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    # output of 10@70 70

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=100, kernel_size=(5,5), stride=(1,1 )), # output of 100@ 66 66
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))  # output of 100@33 33

        self.classifier = nn.Sequential(
            nn.Linear(108900, num_classes)     # seems like i dont need a softmax function, since it automatically does it with cross entropy loss
        )       # this seems extremely massive, but hey, it is what the network suggested

    def forward(self, x):
        # takes the output, then clones the tensor
        out = self.features1(x)
        out = self.features2(out)

        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        # this was the prev method of getting all the needed fmaps, not needed anymore
        #return out, feature_map1, feature_map2, feature_map3
        return out

#endregion

#region custom pca networks 
'''
this section will detail other networks that we are testing. These networks should increase in layers, and will also increase in parameters, as this reflects the natural growth of a neural network when more layers is added
'''
class custom_1conv(nn.Module):
    # this is the original PCA paper https://ieeexplore.ieee.org/document/9109741
    # this is set to 38 since that is what the paper mentioned i guess
    # this one assumes input of size 1x100x100

    # this one is furthur modified that that it returns the feature maps for each of the conv layers i guess
    def __init__(self, num_classes: int = 38, in_channels: int = 1):
        super(custom_1conv, self).__init__()
        '''
        design notes:
            make sure that each of the features are separated cleanly so that they are each convolutional blocks 
            This is so that I can isolate and generate the PCA easier 

            Assume the input is 100x100 pix
        '''
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=(5,5), stride=(2,2 )),    #output of 10@48 48 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    # output of 10@24 24

        self.classifier = nn.Sequential(
            nn.Linear(5760, num_classes)     # seems like i dont need a softmax function, since it automatically does it with cross entropy loss
        )       # this seems extremely massive, but hey, it is what the network suggested

    def forward(self, x):
        out = self.features1(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class custom_4conv(nn.Module):
    # this is the original PCA paper https://ieeexplore.ieee.org/abstract/document/8376025
    # this is set to 38 since that is what the paper mentioned i guess
    # this one assumes input of size 1x100x100

    # this one is furthur modified that that it returns the feature maps for each of the conv layers i guess
    def __init__(self, num_classes: int = 38, in_channels: int = 1):
        super(custom_4conv, self).__init__()
        '''
        design notes:
            make sure that each of the features are separated cleanly so that they are each convolutional blocks 
            This is so that I can isolate and generate the PCA easier 

            Assume the input is 100x100 pix
        '''
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5,5), stride=(1,1 )),    #output of 10@96 96  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    # output of 10@48 48

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1,1 )),  # 12@ 44 44
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    #12@22 22
        
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1,1 )),   #24@20 20
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )    #24@20 20

        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),  #16@18 18
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(5184, num_classes)     # seems like i dont need a softmax function, since it automatically does it with cross entropy loss
        )

    def forward(self, x):
        # takes the output, then clones the tensor
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)

        out = torch.flatten(out, 1) 
        out = self.classifier(out)
        return out


class custom_6conv(nn.Module):
    def __init__(self, num_classes: int = 38, in_channels: int = 1):
        super(custom_6conv, self).__init__()
        '''
        design notes:
            make sure that each of the features are separated cleanly so that they are each convolutional blocks 
            This is so that I can isolate and generate the PCA easier 

            Assume the input is 100x100 pix
        '''
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5,5), stride=(1,1 )),    # Output of 16@96x96  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    # Output of 16@48x48

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1,1 )),  # Output of 32@44x44
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    # Output of 32@22x22
        
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1,1 )),   # Output of 32@20x20
            nn.ReLU()
        )

        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),  # Output of 64@18x18
            nn.ReLU(),
        )
        
        self.features5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),  # Output of 64@16 16
            nn.ReLU()
        )

        self.features6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),  # Output of 16@14
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(3136, num_classes)  # Input features = 16*5*5 = 400
        )

    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = self.features5(out)
        out = self.features6(out)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class custom_8conv(nn.Module):
    def __init__(self, num_classes: int = 38, in_channels: int = 1):
        super(custom_8conv, self).__init__()
        '''
        design notes:
            make sure that each of the features are separated cleanly so that they are each convolutional blocks 
        '''
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5,5), stride=(1,1)),    # Output of 16@96x96  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    # Output of 16@48x48

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1,1)),  # Output of 32@44x44
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))    # Output of 32@22x22
        
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1,1)),   # Output of 32@20x20
            nn.ReLU()
        )

        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),  # Output of 64@18x18
            nn.ReLU()
        )
        
        self.features5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),  # Output of 64@16x16
            nn.ReLU()
        )

        self.features6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),  # Output of 128@14x14
            nn.ReLU()
        )

        self.features7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),  # Output of 128@12x12
            nn.ReLU()
        )

        self.features8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),  # Output of 16@10x10
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1600, num_classes)  # Input features = 16*10*10 = 1600
        )

    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = self.features5(out)
        out = self.features6(out)
        out = self.features7(out)
        out = self.features8(out)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


#endregion

#region available networks 



#region alexnet
# referenced from https://pytorch.org/hub/pytorch_vision_alexnet/
# will include its specific transform as well so that it fits nicely 
def alex_net_model(pretrained=True):
    '''
    pretrained defaults to true here
    '''
    # this function loads in, and provides alexnet 
    import torch
    print(f'pretrained is {pretrained}')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained)
    return model

def alex_net_transform_from_tensor():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def alex_net_transform():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

#endregion

#region resnet
# taken from https://pytorch.org/hub/pytorch_vision_resnet/

def resnet_model(resnet_version='resnet50', pretrained=True):
    '''
    pretrained defaults to true,
    version defaults to resenet 50
    
    full versions are here 
    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
    '''
    model = model = torch.hub.load('pytorch/vision:v0.10.0', resnet_version, pretrained=pretrained)
    return model

def resnet_transform_from_tensor():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def resnet_transform():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

#endregion

#region vgg
# https://pytorch.org/vision/stable/models/vgg.html
def vgg_model(vgg_version='vgg19', pretrained=True):
    '''
    pretrained defaults to True,
    version defaults to vgg19
    
    full versions are here:
    vgg11
    vgg11_bn
    vgg13
    vgg13_bn
    vgg16
    vgg16_bn
    vgg19
    vgg19_bn
    '''
    model = torch.hub.load('pytorch/vision:v0.10.0', vgg_version, pretrained=pretrained)
    return model

def vgg_transform_from_tensor():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def vgg_transform():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform
#endregion

#region inceptionv3
#https://pytorch.org/hub/pytorch_vision_inception_v3/
def inceptionV3_model(pretrained=True):
    '''
    pretrained defaults to True,
    '''
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=pretrained)
    return model

def inceptionV3_transform_from_tensor():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def inceptionV3_transform():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform
#endregion

#region inceptionresnet
# taken from https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/blob/master/model/inception_resnet_v2.py

import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, 32, 3, stride=2, padding=0, bias=False), # 149 x 149 x 32
            Conv2d(32, 32, 3, stride=1, padding=0, bias=False), # 147 x 147 x 32
            Conv2d(32, 64, 3, stride=1, padding=1, bias=False), # 147 x 147 x 64
            nn.MaxPool2d(3, stride=2, padding=0), # 73 x 73 x 64
            Conv2d(64, 80, 1, stride=1, padding=0, bias=False), # 73 x 73 x 80
            Conv2d(80, 192, 3, stride=1, padding=0, bias=False), # 71 x 71 x 192
            nn.MaxPool2d(3, stride=2, padding=0), # 35 x 35 x 192
        )
        self.branch_0 = Conv2d(192, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(192, 48, 1, stride=1, padding=0, bias=False),
            Conv2d(48, 64, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(192, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(192, 64, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv2d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.conv = nn.Conv2d(128, 320, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Conv2d(128, 160, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(160, 192, (7, 1), stride=1, padding=(3, 0), bias=False)
        )
        self.conv = nn.Conv2d(384, 1088, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduciton_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduciton_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 384, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=1, padding=1, bias=False),
            Conv2d(288, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 224, (1, 3), stride=1, padding=(0, 1), bias=False),
            Conv2d(224, 256, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(448, 2080, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=3, classes=1000, k=256, l=256, m=384, n=384):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduciton_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def inception_resnetv2_model():
    return Inception_ResNetv2()

def inception_resnetv2_transform_from_tensor():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def inception_resnetv2_transform():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

#endregion

#endregion