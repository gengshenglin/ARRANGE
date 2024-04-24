import torchvision
from torchvision import transforms

class DataLoader(object):

    def __init__(self, root: str, download: bool = True):
        self.root = root
        self.download = download
        self.data = None


    def load_horizontal_data(self, train: bool = True, resize: list[int] = None):
        trans = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
        if resize != None:
            trans.append(transforms.Resize(size=resize))
        cifar10 = torchvision.datasets.CIFAR10(
            root=self.root,
            download=self.download,
            train=train,
            transform=transforms.Compose(trans)
        )
        self.data = cifar10
        return cifar10
