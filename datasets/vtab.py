import os
from torchvision.datasets.folder import ImageFolder, default_loader


class VTABDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, is_tuning=True, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = target_transform
        self.transform = transform
        if is_tuning:
            train_list_path = os.path.join(self.dataset_root, 'train800.txt')
            test_list_path = os.path.join(self.dataset_root, 'val200.txt')
        else:
            train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
            test_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))