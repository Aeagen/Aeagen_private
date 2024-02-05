# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import pickle
import pickletools
import re
from typing import List, Optional, Tuple, Union, io
import io


import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from torch_utils import gen_utils
#----------------------------------------------------------------------------


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------
def load_g_ema(network_pkl):
    with open(network_pkl, 'rb') as handle:
        d_starts = -1
        g_ema_starts = -1
        for i, op in enumerate(pickletools.genops(handle)):
            if op[0].name == "SHORT_BINUNICODE":
                if op[1] == 'G_ema':
                    g_ema_starts = op[2]
                elif op[1] == 'D':
                    d_starts = op[2]
        assert d_starts >= 0 and g_ema_starts >= 0

    with open(network_pkl, 'rb') as handle:
        bs = handle.read()
        bs = bs[:d_starts] + bs[g_ema_starts:]
        obj = pickle.Unpickler(io.BytesIO(bs)).load()

    return obj['G_ema']


def generate_images(
    seed: List[int],
    class_idx: Optional[int]
):
    network_pkl = 'prenetwork_weight/radimagegan64x64.pkl'
    seeds =parse_range(seed)
    batch_sz = 1
    truncation_psi = 0.7
    centroids_path = None
    noise_mode = 'const'
    outdir = 'out'
    translate = parse_vec2('0,0')
    rotate = 0

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = load_g_ema(network_pkl)
        G = G.eval().requires_grad_(False).to(device)

    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        w = gen_utils.get_w_from_seed(G, batch_sz, device, truncation_psi, seed=seed,
                                      centroids_path=centroids_path, class_idx=class_idx)
        img = gen_utils.w_to_img(G, w, to_np=True)

if __name__ == '__main__':
    generate_images(seed = '10-15',class_idx = 15)