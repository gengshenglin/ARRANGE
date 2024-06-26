import torchvision  # 用于导入数据集
import torchvision.transforms as transforms  # 用于数据集格式转换
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


class DataLoader(object):
    """
    1.root：数据集路径
    2.download：若root路径下没有数据集，是否下载
    """

    def __init__(self, root: str, download: bool = True):
        self.root = root
        self.download = download
        self.data = None

    """
    1.train：True-下载训练集，False-下载测试集
    """

    def load_horizontal_data(self, train: bool = True, resize: list[int] = None):
        trans = []
        trans.append(transforms.ToTensor())
        if resize != None:
            trans.append(transforms.Resize(size=resize))

        # 加载数据集
        mnist = torchvision.datasets.MNIST(
            root=self.root,
            download=self.download,
            train=train,
            transform=transforms.Compose(trans)
        )
        self.data = mnist
        return mnist
