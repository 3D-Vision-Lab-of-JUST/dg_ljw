
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import argparse
from torch import nn
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import torch
from torch.utils import data
from tensorboardX import SummaryWriter
from configs.configs import *
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.autograd import Variable


from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.train_UDA import train_domain_adaptation,print_losses
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss, prob_2_entropy

from dataset.mapillary import MapillaryDataSet
from dataset.synthia import SYNTHIADataSetDepth
from model.deeplabv2_depth import get_deeplab_v2_depth
from configs.configs import cfg, cfg_from_file
from utils.util import *


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    # LOAD ARGS
    args = get_arguments()

    args.cfg="configs/configs_s2c/dg.yml"
    print("Called with args:")
    print(args)

    assert args.cfg is not None, "Missing cfg file"
    cfg_from_file(args.cfg)

    # auto-generate exp name if not specified
    if cfg.EXP_NAME == "":
        cfg.EXP_NAME = f"{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}"

    if args.exp_suffix:
        cfg.EXP_NAME += f"_{args.exp_suffix}"
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == "":
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == "":
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(
                cfg.EXP_ROOT_LOGS, "tensorboard", cfg.EXP_NAME
            )
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ""
    print("Using config:")
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)


    # LOAD SEGMENTATION NET
    assert osp.exists(
        cfg.TRAIN.RESTORE_FROM
    ), f"Missing init model {cfg.TRAIN.RESTORE_FROM}"
    if cfg.TRAIN.MODEL == "DeepLabv2_depth":
        model = get_deeplab_v2_depth(
            num_classes=cfg.NUM_CLASSES,
            multi_level=cfg.TRAIN.MULTI_LEVEL
        )
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if "DeepLab_resnet_pretrained_imagenet" in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split(".")
                if not i_parts[1] == "layer5":
                    new_params[".".join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print("Model loaded")

    # DATALOADERS
    source_dataset = SYNTHIADataSetDepth(
        root=cfg.DATA_DIRECTORY_SOURCE,
        list_path=cfg.DATA_LIST_SOURCE,
        set=cfg.TRAIN.SET_SOURCE,
        num_classes=cfg.NUM_CLASSES,
        max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
        crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
        mean=cfg.TRAIN.IMG_MEAN,
        use_depth=cfg.USE_DEPTH,
    )
    #train_source_sample = DistributedSampler(source_dataset)
    source_loader = data.DataLoader(
        source_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        # sampler = train_source_sample,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    if cfg.TARGET == 'Cityscapes':
        target_dataset = CityscapesDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN
        )
    elif cfg.TARGET == 'Mapillary':
        target_dataset = MapillaryDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=True
        )
    else:
        raise NotImplementedError(f"Not yet supported dataset {cfg.TARGET}")
    # train_target_sample = DistributedSampler(target_dataset)
    target_loader = data.DataLoader(
        target_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        # sampler = train_target_sample,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, "train_cfg.yml"), "w") as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    if cfg.USE_DEPTH:
        train_domain_adaptation_with_depth(model, source_loader, target_loader, cfg)
    else:
        train_domain_adaptation(model, source_loader, target_loader, cfg)

def train_dada( model, trainloader, targetloader, cfg):
    """ UDA training with dada
    """
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()

    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(
        model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
        #model.module.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,  #dada = cfg.TRAIN.WEIGHT_DECAY
   )

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(
        d_main.parameters(),
        #d_main.module.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE_D,
        betas=(0.9, 0.99)
    )

    # interpolate output segmaps
    interp = nn.Upsample(
        size=(input_size_source[1], input_size_source[0]),
        mode="bilinear",
        align_corners=True,
    )
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP+1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, depth, _, _ = batch
        _, pred_src_main, pred_depth_src_main = model(images_source.cuda(device))
        # _, pred_src_main, pred_depth_src_main = model(images_source.cuda(local_rank))
        pred_src_main = interp(pred_src_main)
        pred_depth_src_main = interp(pred_depth_src_main)
        loss_depth_src_main = loss_calc_depth(pred_depth_src_main, depth, device)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = ( cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_DEPTH_MAIN * loss_depth_src_main)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        _, pred_trg_main, pred_depth_trg_main = model(images.cuda(device))
        #_, pred_trg_main, pred_depth_trg_main = model(images.cuda(local_rank))
        pred_trg_main = interp_target(pred_trg_main)
        pred_depth_trg_main = interp_target(pred_depth_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_depth_trg_main * pred_depth_trg_main)
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        pred_src_main = pred_src_main.detach()
        pred_depth_src_main = pred_depth_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)) * pred_depth_src_main)
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main
        loss_d_main.backward()

        # train with target
        pred_trg_main = pred_trg_main.detach()
        pred_depth_trg_main = pred_depth_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_depth_trg_main * pred_depth_trg_main)
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_main.step()

        current_losses = {
            "loss_seg_src_main": loss_seg_src_main,
            "loss_depth_src_main": loss_depth_src_main,
            "loss_adv_trg_main": loss_adv_trg_main,
            "loss_d_main": loss_d_main,
        }
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0 :
            print("taking snapshot ...")
            print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f"model_{i_iter}.pth")
            torch.save(d_main.state_dict(), snapshot_dir / f"model_{i_iter}_D_main.pth")
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()





def train_domain_adaptation_with_depth(model, trainloader, targetloader, cfg):
    assert cfg.TRAIN.DA_METHOD in {"dg"}, "Not yet supported DA method {}".format(cfg.TRAIN.DA_METHOD)
    train_dada(model, trainloader, targetloader, cfg)

if __name__ == "__main__":
    main()