#!/usr/bin/env python3

import argparse
from data import build_rocstories_dataset, build_github_dataset
from model import train_model, generate_from_model

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cogents: Controllable Generation of Text from Slices')
    parser.add_argument('--build-dataset', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--rocstories', action='store_true')
    parser.add_argument('--github', action='store_true')
    parser.add_argument('--language', help='Java, Python, or Haskell', default = 'Python')
    parser.add_argument('--model', help='Path to trained model.')
    parser.add_argument('--output', help='Path to output file.')
    parser.add_argument('--dataset', help='Path to pre-processed dataset.')
    parser.add_argument('--devices', help='Torch device to run on.', default= 'cpu')
    parser.add_argument('--transformer', help='which transformer to use', default= 'vanilla')
    parser.add_argument('--strat', help='which training strategy to use', default= 'ddp')
    parser.add_argument('--n_head', help='number of heads', default= 12, type = int)
    parser.add_argument('--n_layer', help='number of layers', default= 12, type = int)
    parser.add_argument('--eval_perplexity', help='compute average perpexity on test set', action='store_true')

    opt = parser.parse_args()

    if opt.build_dataset:
        if opt.rocstories:
            build_rocstories_dataset(opt.output)
        elif opt.github:
            build_github_dataset(opt.output, opt.language)
        else:
            raise ValueError('Specify one of --rocstories or --github')
    elif opt.train:
        train_model(
                opt.dataset, 
                None if opt.devices == 'cpu' else list(map(int, opt.devices.split(','))),
                opt.transformer,
                opt.output,
                opt.strat,
                opt.n_head,
                opt.n_layer)
    elif opt.test:
        generate_from_model(opt.dataset, opt.model, opt.transformer,[] if opt.devices == 'cpu' else int(opt.devices), opt.eval_perplexity)
