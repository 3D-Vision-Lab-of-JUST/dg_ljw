import argparse
import os
import os.path as osp
import pprint
import random
import warnings


import sys
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

import numpy as np
import yaml
import torch
from torch.utils import data

from advent.utils.func import loss_calc, bce_loss
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.model.discriminator import get_fc_discriminator
from advent.dataset.cityscapes import CityscapesDataSet as CityscapesDataSet_hard
from dataset.cityscapes_intra import CityscapesDataSet as CityscapesDataSet_easy
from configs.configs_intra import cfg, cfg_from_file
from advent.utils.func import prob_2_entropy

import sys
sys.path.append("..")
from model.deeplabv2_depth import get_deeplab_v2_depth
from dada.dataset.mapillary import MapillaryDataSet as MapillaryDataSet_hard

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default='./intrada.yml', # None
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz_every_iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()

def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint)
    saved_state_dict = saved_state_dict['model_state_dict']
    model.load_state_dict(saved_state_dict)
    # model.eval()
    # model.cuda(device)

def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    # pdb.set_trace()

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    print('Using config:')
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

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET

    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2_depth(num_classes=cfg.NUM_CLASSES, multi_level=False)
        restore_from = '/home/ailab/ailab/LJW/our_code/pretrained/model_13000_4_46.09.pth'
        # model.load_state_dict(torch.load(restore_from))
        load_checkpoint_for_evaluation(model, restore_from)
        # saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        # if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
        #     new_params = model.state_dict().copy()
        #     for i in saved_state_dict:
        #         i_parts = i.split('.')
        #         if not i_parts[1] == 'layer5':
        #             new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        #     model.load_state_dict(new_params)
        # else:
        #     model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')

    # DATALOADERS
    # pdb.set_trace()
    easy_dataset = CityscapesDataSet_easy(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                 mean=cfg.TRAIN.IMG_MEAN)
    easy_loader = data.DataLoader(easy_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=False,
                                    worker_init_fn=_init_fn)

    # easy_dataset = MapillaryDataSet_easy(root=cfg.DATA_DIRECTORY_SOURCE,
    #                              list_path=cfg.DATA_LIST_SOURCE,
    #                              max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
    #                              crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
    #                              mean=cfg.TRAIN.IMG_MEAN)
    #
    # easy_loader = data.DataLoader(easy_dataset,
    #                                 batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
    #                                 num_workers=cfg.NUM_WORKERS,
    #                                 shuffle=True,
    #                                 pin_memory=False,
    #                                 worker_init_fn=_init_fn)

    # pdb.set_trace()
    # CityscapesDataSet_hard
    hard_dataset = CityscapesDataSet_hard(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)
    hard_loader = data.DataLoader(hard_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=False,
                                    worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # pdb.set_trace()
    train_domain_adaptation(model, easy_loader, hard_loader, cfg)


def train_domain_adaptation(model, trainloader, targetloader, cfg):

    if cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

# def load_checkpoint_for_evaluation(model, checkpoint, device):
#     saved_state_dict = torch.load(checkpoint)
#     model.load_state_dict(saved_state_dict)
#     model.eval()
#     model.cuda(device)


def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    # pdb.set_trace()
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # restore_from = cfg.TRAIN.RESTORE_FROM_aux
    # print("Load Discriminator:", restore_from)
    # load_checkpoint_for_evaluation(d_aux, restore_from, device)


    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # restore_from = cfg.TRAIN.RESTORE_FROM_main
    # print("Load Discriminator:", restore_from)
    # load_checkpoint_for_evaluation(d_main, restore_from, device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        # debug:
        # labels=labels.numpy()
        # from matplotlib import pyplot as plt
        # import numpy as np
        # plt.figure(1), plt.imshow(labels[0]), plt.ion(), plt.colorbar(), plt.show()
        pred_src_aux, pred_src_main,_ = model(images_source.cuda(device))
        #_, pred_src_main, _ = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
       # print("aaaaaaaa:", labels.size())
       # print("bbbbbbbbbbb:", pred_src_main.size())
        # labels[labels <= 0] = 1
        # labels[labels > 7] = 7
        loss = 0
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)

        # pdb.set_trace()
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        #loss =  loss_seg_src_main
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_src_aux, pred_trg_main, _ = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        #loss = loss_seg_src_main
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        # for param in d_aux.parameters():
        #     param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_src_aux = pred_src_aux.detach()
        #     d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
        #     loss_d_aux = bce_loss(d_out_aux, source_label)
        #     loss_d_aux = loss_d_aux / 2
        #     loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        # if cfg.TRAIN.MULTI_LEVEL:
        #     optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

if __name__ == '__main__':
    main()
