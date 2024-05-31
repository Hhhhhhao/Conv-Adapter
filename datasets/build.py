from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

from .vtab import VTABDataset
from .fgvc import CUB2011, NABirds, DogsDataset, CarsDataset
from .fewshot import FewShotDataset


def build_transform(args, is_train):
    if is_train:
        if args.data == 'vtab-1k':
            if args.dataset in ['clevr_dist', 'clevr_count', 'dsprites_loc', 'dsprites_ori', 'dmlab', 'smallnorb_azi']:
                hflip = 0.0
            else:
                hflip = 0.5
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                hflip=hflip,
                interpolation='bicubic',
                color_jitter=None,
                # manually set aa to None
                auto_augment=None,
                re_prob=0.0,
            )
            if args.dataset in ['clevr_count', 'dmlab', 'clever_dist', 'kitti', 'smallnorb_azi']:
                size = int(args.input_size / args.crop_ratio)
                transform.transforms[0] = transforms.Compose([
                    transforms.Resize((size,size), interpolation=3),
                    transforms.RandomCrop(args.input_size),
                ])
            elif args.dataset in ['dsprites_loc']:
                transform.transforms[0] = transforms.Compose([
                    transforms.Resize((args.input_size,args.input_size), interpolation=3),
                ])
        elif args.data == 'fgvc' or args.data == 'clip' or args.data == 'few-shot':
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                scale=(0.22, 1.0),
                # manually set aa to None
                auto_augment=None,
                interpolation='bicubic',
                re_prob=0.0,
            )
        else:
            t = []
            size = int(args.input_size / args.crop_ratio)
            t.append(
                transforms.Resize(size, interpolation=3)  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))
            t.append(transforms.RandomHorizontalFlip())
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            transform = transforms.Compose(t)

        return transform

    t = []
    size = int(args.input_size / args.crop_ratio)
    if args.dataset == 'dsprites_loc':
        t.append(
            transforms.Resize((args.input_size,args.input_size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
        )
    else:
        t.append(
            transforms.Resize((size,size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(args, is_train=True):
    transform = build_transform(args, is_train)

    # vtab-1k datasets
    if args.data == 'vtab-1k':
        if args.dataset == 'clevr_count':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 8
        elif args.dataset == 'diabetic_retinopathy':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 5
        elif args.dataset == 'dsprites_loc':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 16
        elif args.dataset == 'dtd':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 47
        elif args.dataset == 'kitti':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 4
        elif args.dataset == 'oxford_pet':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 37
        elif args.dataset == 'resisc45':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 45
        elif args.dataset == 'smallnorb_ele':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 9
        elif args.dataset == 'svhn':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 10
        elif args.dataset == 'cifar100':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 100
        elif args.dataset == 'clevr_dist':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 6
        elif args.dataset == 'caltech101':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 102
        elif args.dataset == 'dmlab':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 6
        elif args.dataset == 'dsprites_ori':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 16
        elif args.dataset == 'eurosat':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 10
        elif args.dataset == 'oxford_flowers102':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 102
        elif args.dataset == 'patch_camelyon':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 2
        elif args.dataset == 'smallnorb_azi':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 18
        elif args.dataset == 'sun397':
            dataset = VTABDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 397
        else:
            raise NotImplementedError
    elif args.data == 'fgvc':
        # fgvc datasets
        if args.dataset == 'cub2011':
            dataset = CUB2011(args.data_path, is_train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 200
        elif args.dataset == 'dogs':
            dataset = DogsDataset(root=args.data_path, train=is_train, cropped=False, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 120
        elif args.dataset == 'cars':
            dataset = CarsDataset(root=args.data_path, is_train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 197
        elif args.dataset == 'nabirds':
            dataset = NABirds(root=args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning)
            nb_classes = 700
        else:
            raise NotImplementedError
    elif args.data == 'clip':
        if args.dataset == 'cifar100':
            dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
            nb_classes = 100
        elif args.dataset == 'cifar10':
            dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
            nb_classes = 10
        elif args.dataset == 'flowers':
            dataset = datasets.Flowers102(args.data_path, split='train' if is_train else 'test', transform=transform, download=True)
            nb_classes = 102
        elif args.dataset == 'food':
            dataset = datasets.Food101(args.data_path, split='train' if is_train else 'test', transform=transform,download=True)
            nb_classes = 101
        elif args.dataset == 'eurasat':
            dataset = datasets.EuroSAT(args.data_path, split='train' if is_train else 'test', transform=transform,download=True)
            nb_classes = 101
    elif args.data == 'few-shot':
        dataset = FewShotDataset(args.data_path, train=is_train, transform=transform, is_tuning=args.is_tuning, shot=args.fs_shot, seed=args.seed)
        if args.dataset == 'fgvc_aircraft':
            nb_classes = 100
        elif args.dataset == 'oxford_flowers102':
            nb_classes = 102
        elif args.dataset == 'food101':
            nb_classes = 101
        elif args.dataset == 'stanford_cars':
            nb_classes = 196
        elif args.dataset == 'oxford_pets':
            nb_classes = 37
    else:
        raise NotImplementedError

    return dataset, nb_classes