import torchvision


def MyResNet56():
    return torchvision.models.resnet56(pretrained=True)
