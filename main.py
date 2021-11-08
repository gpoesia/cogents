#!/usr/bin/env python3

import argparse
from data import build_rocstories_dataset
from model import train_model, generate_from_model

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cogents: Controllable Generation of Text from Slices')
    parser.add_argument('--build-dataset', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--rocstories', action='store_true')
    parser.add_argument('--github', action='store_true')
    parser.add_argument('--output', help='Path to output file.')
    parser.add_argument('--dataset', help='Path to pre-processed dataset.')
    parser.add_argument('--devices', help='Torch device to run on.', default= 'cpu')
    parser.add_argument('--transformer', help='which transformer to use', default= 'vanilla')

    opt = parser.parse_args()

    if opt.build_dataset:
        if opt.rocstories:
            build_rocstories_dataset(opt.output)
        else:
            raise ValueError('Specify one of --rocstories or --github')
    elif opt.train:
        train_model(
                opt.dataset, 
                [] if opt.devices == 'cpu' else list(map(int, opt.devices.split(','))),
                opt.transformer,
                opt.output)
    elif opt.test:
        generate_from_model(opt.dataset, torch.device(opt.devices))
