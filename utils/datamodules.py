from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100, SVHN, STL10
from torchvision import transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str='./data/', batch_size=256, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.batch_size = batch_size
        self.num_workerd=num_workers
 
    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
 
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.CIFAR10_train = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            self.CIFAR10_val = CIFAR10(self.data_dir, train=False, transform=self.transform_test)
 
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.CIFAR10_test = CIFAR10(self.data_dir, train=False, transform=self.transform_test)
 
    def train_dataloader(self):
        return DataLoader(self.CIFAR10_train, batch_size=self.batch_size, num_workers=self.num_workerd)

    def val_dataloader(self):
        return DataLoader(self.CIFAR10_val, batch_size=self.batch_size, num_workers=self.num_workerd)
 
    def test_dataloader(self):
        return DataLoader(self.CIFAR10_test, batch_size=self.batch_size, num_workers=self.num_workerd)


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str='./data/', batch_size=256, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        self.dims = (3, 32, 32)
        self.num_classes = 100
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.CIFAR100_train = CIFAR100(self.data_dir, train=True, download=True, transform=self.transform_train)
            self.CIFAR100_val = CIFAR100(self.data_dir, train=False, download=True, transform=self.transform_test)
        
        if stage == 'test' or stage is None:
            self.CIFAR100_test = CIFAR100(self.data_dir, train=False, transform=self.transform_test)
    
    def train_dataloader(self):
        return DataLoader(self.CIFAR100_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.CIFAR100_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.CIFAR100_test, batch_size=self.batch_size, num_workers=self.num_workers)


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str='./data/', batch_size=256, num_workers=8, shuffle=False):
        super().__init__()
        self.data_dir = data_dir
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dims = (28, 28)
        self.num_classes = 10
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.FashionMNIST_train = FashionMNIST(self.data_dir, train=True, download=True, transform=self.transform_train)
            self.FashionMNIST_val = FashionMNIST(self.data_dir, train=False, download=True, transform=self.transform_test)
        
        if stage == 'test' or stage is None:
            self.FashionMNIST_test = FashionMNIST(self.data_dir, train=False, transform=self.transform_test)
    
    def train_dataloader(self):
        return DataLoader(self.FashionMNIST_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.FashionMNIST_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.FashionMNIST_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)


class SVHNDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str='./data/', batch_size=256, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.SVHN_train = SVHN(self.data_dir, split='train', download=True, transform=self.transform_train)
            self.SVHN_val = SVHN(self.data_dir, split='test', download=True, transform=self.transform_test)
        
        if stage == 'test' or stage is None:
            self.SVHN_test = SVHN(self.data_dir, split='test', transform=self.transform_test)
    
    def train_dataloader(self):
        return DataLoader(self.SVHN_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.SVHN_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.SVHN_test, batch_size=self.batch_size, num_workers=self.num_workers)


class STL10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str='./data/', batch_size=256, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.dims = (3, 96, 96)
        self.num_classes = 10
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.STL10_train = STL10(self.data_dir, split='train', download=True, transform=self.transform_train)
            self.STL10_val = STL10(self.data_dir, split='test', download=True, transform=self.transform_test)
        
        if stage == 'test' or stage is None:
            self.STL10_test = STL10(self.data_dir, split='test', transform=self.transform_test)
    
    def train_dataloader(self):
        return DataLoader(self.STL10_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.STL10_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.STL10_test, batch_size=self.batch_size, num_workers=self.num_workers)