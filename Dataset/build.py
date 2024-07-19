from torchvision import transforms
from .img_transfroms import *


def build_trainset(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if cfg.dataset == 'culane':
        from .culane_dataset import CULaneTrSet
        trainset = CULaneTrSet(cfg=cfg, transforms=transform)

    elif cfg.dataset == 'llamas':
        from .llamas_dataset import LLAMASTrSet
        trainset = LLAMASTrSet(cfg=cfg, transforms=transform)

    elif cfg.dataset == 'tusimple':
        from .tusimple_dataset import TuSimpleTrSet
        trainset = TuSimpleTrSet(cfg=cfg, transforms=transform)

    elif cfg.dataset == 'dlrail':
        from .dlrail_dataset import DLRailTrSet
        trainset = DLRailTrSet(cfg=cfg, transforms=transform)

    elif cfg.dataset == 'curvelanes':
        from .curvelanes_dataset import CurveLanesTrSet
        trainset = CurveLanesTrSet(cfg=cfg, transforms=transform)
        
    else:
        trainset = None
    return trainset


def build_testset(cfg):
    pre_transforms = transforms.Compose([
        Resize(cfg.img_h, cfg.img_w),
        transforms.ToTensor(),
    ])

    if cfg.dataset == 'culane':
        from .culane_dataset import CULaneTsSet
        testset = CULaneTsSet(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'llamas':
        from .llamas_dataset import LLAMASTsSet
        testset = LLAMASTsSet(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'tusimple':
        from .tusimple_dataset import TusimpleTsSet
        testset = TusimpleTsSet(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'dlrail':
        from .dlrail_dataset import DLRailTsSet
        testset = DLRailTsSet(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'curvelanes':
        from .curvelanes_dataset import CurveLanesTsSet
        testset = CurveLanesTsSet(cfg=cfg, transforms=pre_transforms)

    else:
        testset=None
    return testset


def build_viewtrset(cfg):
    pre_transforms = transforms.Compose([
        Resize(cfg.img_h, cfg.img_w),
        transforms.ToTensor(),
    ])

    if cfg.dataset == 'culane':
        from .culane_dataset import CULaneTrSetView
        testset = CULaneTrSetView(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'llamas':
        from .llamas_dataset import LLAMASTrSetView
        testset = LLAMASTrSetView(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'tusimple':
        from .tusimple_dataset import TuSimpleTrSetView
        testset = TuSimpleTrSetView(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'dlrail':
        from .dlrail_dataset import DLRailTrSetView
        testset = DLRailTrSetView(cfg=cfg, transforms=pre_transforms)

    elif cfg.dataset == 'curvelanes':
        from .curvelanes_dataset import CurveLanesTrSetView
        testset = CurveLanesTrSetView(cfg=cfg, transforms=pre_transforms)

    else:
        testset=None
    return testset
