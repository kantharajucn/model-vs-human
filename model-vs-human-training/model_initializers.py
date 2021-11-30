import torch.nn as nn
from model_initialization_registry import register


@register
def resnet18(model):
    print("initializing resnet18 model")
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 16)
    return model


@register
def resnet50(model):
    print("initializing resnet50 model")
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 16)
    return model


@register
def alexnet(model):
    print("initializing alexnet model")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 16)
    return model


@register
def vgg(model):
    print("initializing vgg model")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 16)
    return model


@register
def squeezenet1_1(model):
    print("initializing squeezenet1_1 model")
    for param in model.parameters():
        param.requires_grad = False
    print(model.classifier)
    model.classifier[1] = nn.Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1))
    return model


@register
def densenet121(model):
    print("initializing Densenet model")
    for param in model.parameters():
        param.requires_grad = False
    print(model.classifier)
    model.classifier = nn.Linear(1024, 16)
    return model

