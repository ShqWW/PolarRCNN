import os
import warnings
import math
import numpy as np
import random
import argparse
import torch
from torch import optim
from utils.dataloaderx import DataLoaderX
from tools.multi_gpu import *
from tools.loss_rec import LossRecorder
from tools.get_config import get_cfg
from Models.build import build_model
from Dataset.build import build_trainset
from Loss.overallloss import OverallLoss

warnings.filterwarnings("ignore", category=RuntimeWarning)
import time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='train-config')
    #device setting
    parser.add_argument('--is_multigpu', default=0, type=int)
    parser.add_argument('--is_gpu', default=1, type=bool)
    parser.add_argument('--gpu_no', default=5, type=int)
    #loss recorder setting
    parser.add_argument('--use_tensorboard', default=False, type=bool)
    parser.add_argument('--iter_display', default=80, type=int)
    #training setting
    parser.add_argument('--cfg', default='./Config/config.py', type=str)
    parser.add_argument('--save_path', default='work_dir/ckpt', type=str)

    args = parser.parse_args()
    return args

def set_random_seed(seed: int) -> None:
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True

def get_dataloader(dataset, is_multigpu=False):
    if is_multigpu:
        sampler = DistributedSampler(dataset)
        batch_size_train = int(cfg.batch_size / get_world_size())
        dataloader = DataLoaderX(dataset, batch_size=batch_size_train,
                                                 num_workers=8,
                                                 collate_fn=dataset.collate_fn,
                                                 sampler=sampler)
        return dataloader, sampler
    else:
        dataloader = DataLoaderX(dataset, batch_size=cfg.batch_size,
                                                 num_workers=16,
                                                 shuffle=1,
                                                 collate_fn=dataset.collate_fn,
                                                 drop_last=True)
        return dataloader, None

def get_model(net, is_gpu=False, is_multigpu=False):
    if is_gpu or is_multigpu:
        net.cuda()
    if is_multigpu:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[GPUNO], find_unused_parameters=False)
    return net

def train_one_epoch(epoch, net, trloader, trsampler, optimizer, lr_sche, lossfun, recorder, is_gpu=False, is_multigpu=False):
    net.train()
    if ISMAINPROCESS:
        print('Epoch:', epoch+1)
    for iter, sample_batch in enumerate(trloader):
        if recorder.total_iter >= recorder.max_iter:
            break
        if is_gpu or is_multigpu:
            for key in sample_batch.keys():
                if 'list' in key:
                    sample_batch[key] = [data.cuda() for data in sample_batch[key]]
                else:
                    sample_batch[key] = sample_batch[key].cuda() 
        if is_multigpu:
            trsampler.set_epoch(epoch)
        optimizer.zero_grad()
        pred = net(sample_batch)
        loss, loss_msg = lossfun(pred, sample_batch)
        loss.backward()
        optimizer.step()

        recorder.record(loss_msg)
        if ISMAINPROCESS and (iter+1)%cfg.iter_display==0:
            recorder.display(iter, lr = optimizer.param_groups[0]['lr'])
        lr_sche.step()   # lr_sche by epoch, if you want to update by iter, please put this line in the loop
    return net

def train():
    trset = build_trainset(cfg)
    trloader, trsampler = get_dataloader(trset, MULTIGPU)
    net = build_model(cfg) 
    net = get_model(net, is_gpu=GPU, is_multigpu=MULTIGPU)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr)
    max_iter  = len(trset)//cfg.batch_size*cfg.epoch_num
    if ISMAINPROCESS:
        print('dataset length:', len(trset))
        print('epoch number:', cfg.epoch_num)
    def warm_up_cosine_lr(iteration):
        warm_up = cfg.warmup_iter
        if iteration <= warm_up:
            return iteration / warm_up
        else:
            return 0.5 * (math.cos((iteration - warm_up) / (max_iter - warm_up) * math.pi) + 1)
    
    lr_sche = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_cosine_lr)
    loss_fun = OverallLoss(cfg=cfg)
    if GPU or MULTIGPU:
        loss_fun = loss_fun.cuda()
    recorder = LossRecorder(is_tb=cfg.use_tensorboard, process_valid=ISMAINPROCESS, max_iter=max_iter)
    os.makedirs(cfg.save_path, exist_ok=True)

    for epoch_no in range(cfg.epoch_num):
        net = train_one_epoch(epoch_no, net, trloader, trsampler,\
                              optimizer, lr_sche, loss_fun, recorder, \
                              is_gpu=GPU, is_multigpu=MULTIGPU)
        if ISMAINPROCESS:
            net_weight = net.module.state_dict() if cfg.is_multigpu else net.state_dict() 
        if ISMAINPROCESS and epoch_no%2==0:
            save_weight = os.path.join(cfg.save_path, 'para_{}.pth'.format(str(epoch_no)))
            torch.save(net_weight, save_weight)
    if ISMAINPROCESS:
        save_weight = os.path.join(cfg.save_path, 'para_{}.pth'.format(str(cfg.epoch_num-1)))
        torch.save(net_weight, save_weight) 
        
cfg = get_cfg(parse_args())
MULTIGPU = cfg.is_multigpu   # use multiple gpu or not
GPU = cfg.is_gpu    # use single gpu or not
GPUNO = cfg.gpu_no  # single gpu no
ISMAINPROCESS = True    

if __name__ == '__main__':
    set_random_seed(cfg.random_seed)
    if MULTIGPU:
        GPUNO = int(os.environ["LOCAL_RANK"])
        device_ids = range(torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl")
    ISMAINPROCESS = (cfg.is_multigpu==0) | (GPUNO == 0)
    torch.cuda.set_device(GPUNO)
    if ISMAINPROCESS:
        print(cfg)
        print("gpu_no:", GPUNO)
    train()