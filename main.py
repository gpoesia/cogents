#!/usr/bin/env python3

import argparse
from data import build_rocstories_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cogents: Controllable Generation of Text from Slices')
    parser.add_argument('--build-dataset', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--rocstories', action='store_true')
    parser.add_argument('--github', action='store_true')
    parser.add_argument('--output', help='Path to output file.')

    opt = parser.parse_args()

    if opt.build_dataset:
        if opt.rocstories:
            build_rocstories_dataset(opt.output)
        else:
            raise ValueError('Specify one of --rocstories or --github')
