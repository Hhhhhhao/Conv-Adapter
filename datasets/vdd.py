import os
import os.path
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, dataset, is_train=True, is_tunning=False,
                       transform=None, target_transform=None,
                       labels=None ,imgs=None,loader=pil_loader):
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_train = is_train
        self.is_tuning = is_tunning
        self.imgs, self.labels, self.num_classes = self.load_data()
    

    def load_data(self):
        from pycocotools.coco import COCO
        
        if self.is_train:
            coco = COCO(os.path.join(self.root, 'decathlon-1.0/annotations', '{dataset}_train.json'))
        else:
            if self.is_tunning:
                coco = COCO(os.path.join(self.root, 'decathlon-1.0/annotations', '{dataset}_val.json'))
            else:
                coco = COCO(os.path.join(self.root, 'decathlon-1.0/annotations', '{dataset}_test_stripped.json'))

        imgIds = coco.getImgIds()
        annIds = coco.getAnnIds(imgIds=imgIds)
        anno = coco.loadAnns(annIds)
        images = coco.loadImgs(imgIds) 
        timgnames = [img['file_name'] for img in images]
        timgnames_id = [img['id'] for img in images]
        labels = [int(ann['category_id'])-1 for ann in anno]
        min_lab = min(labels)
        labels = [lab - min_lab for lab in labels]
        max_lab = max(labels)
        imgnames = []
        for j in range(len(timgnames)):
            imgnames.append((self.root + '/' + timgnames[j],timgnames_id[j]))
        return imgnames, labels, int(max_lab+1)


    def __getitem__(self, index):
        path = self.imgs[index][0]
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)