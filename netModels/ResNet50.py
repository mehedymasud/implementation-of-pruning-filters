import torchvision


def MyResNet50():
    return torchvision.models.resnet50(pretrained=True)
