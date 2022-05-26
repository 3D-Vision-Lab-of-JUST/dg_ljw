
import argparse
import torch
import os
import sys

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            if not os.path.exists(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))
                self.file = open(fpath, 'w')
            else:
                self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def loss_calc_depth(pred, label, device):
    n, c, h, w = pred.size()
    assert c == 1

    pred = pred.squeeze()
    label = label.squeeze().cuda(device)

    adiff = torch.abs(pred - label)
    batch_max = 0.2 * torch.max(adiff).item()
    t1_mask = adiff.le(batch_max).float()
    t2_mask = adiff.gt(batch_max).float()
    t1 = adiff * t1_mask
    t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
    t2 = t2 * t2_mask
    return (torch.sum(t1) + torch.sum(t2)) / torch.numel(pred.data)

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("--local_rank",help="local device id",type=int)

    return parser.parse_args()