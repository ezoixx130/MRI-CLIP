# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image


logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    # model.load_state_dict(state_dict, strict=False)
    # logger.info("Pretrained weights found at {}".format(pretrained_weights))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class CsvDataset(Dataset): # added by Hanwen to load csv data
    def __init__(self, input_filename, transform, img_key):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename)

        self.images = df[img_key].tolist()
        self.transform = transform
        print('Done loading {} images in total.'.format(len(self.images)))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        for _ in range(10):
            try:
                img = Image.open(self.images[idx]).convert('RGB')
                img = self.transform(img)
                return img, '-'.join([self.images[idx].split('/')[-3], self.images[idx].split('/')[-1]]).split('.')[0], 1.0
            except:
                idx = np.random.randint(len(self.images))
                continue
        
