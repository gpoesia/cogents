#!/usr/bin/env python3

import csv
from dataclasses import dataclass
from typing import List, Dict
import random
from tqdm import tqdm
import gzip
import json
import re

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

from functools import reduce


@dataclass
class Example:
    context: str
    signal: str
    answer: str


@dataclass
class Dataset:
    train: List[Example]
    val: List[Example]
    test: List[Example]
    tokenizer: Tokenizer


@dataclass
class ShortStory:
    title: str
    lines: str


def train_tokenizer(data):
    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.normalizer = normalizers.BertNormalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()

    trainer = WordPieceTrainer(
        special_tokens=["[UNK]", "[SIG]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"],
        vocab_size=2048,
    )

    tokenizer.train_from_iterator(data, trainer)
    return tokenizer


def train_code_tokenizer(data):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.decoder = decoders.BPE()

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[SIG]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"],
        vocab_size=2048,
    )

    tokenizer.train_from_iterator(data, trainer)
    return tokenizer


def random_split(l, proportions):
    splits = []
    l = l.copy()
    random.shuffle(l)
    i = 0

    for p in proportions:
        j = i + int(p * len(l))
        splits.append(l[i:j])
        i = j

    splits[-1].extend(l[i:])
    return tuple(splits)


def build_rocstories_dataset(output):
    def load_rocstories(path):
        with open(path) as f:
            return [ShortStory(title=row['storytitle'],
                               lines=[row[f'sentence{i}'] for i in range(1, 6)])
                    for row in csv.DictReader(f)]
    rocstories = (load_rocstories('datasets/rocstories_2016.csv') +
                  load_rocstories('datasets/rocstories_2017.csv'))

    def make_examples(stories):
        ex = []

        for s in tqdm(stories):
            for i in range(10):
                for k in range(1, 5):
                    target_line = random.randint(0, len(s.lines) - 1)
                    context_lines = [s.title] + s.lines[:target_line]
                    target_words = s.lines[target_line].split()
                    signal = random.sample(target_words, min(k, len(target_words)))
                    ex.append(Example('\n'.join(context_lines), ' '.join(signal), s.lines[target_line]))
        breakpoint()
        return ex

    print(len(rocstories), 'stories loaded.')

    random.seed('cogents-rocstories')
    (train, val, test) = random_split(rocstories, [0.8, 0.1, 0.1])
    tokenizer_data = [s.title for s in train] + [l for s in train for l in s.lines]
    tokenizer = train_tokenizer(tokenizer_data)

    train_ex, val_ex, test_ex = make_examples(train), make_examples(val), make_examples(test)
    d = Dataset(train_ex, val_ex, test_ex, tokenizer)
    print(len(train_ex), 'training examples.')
    torch.save(d, output)

def split_at_identifier_boundaries(s):
    '''Returns a list of strings obtained by splitting s at identifier boundaries.

    Example: split_at_identifier_boundaries('self.x0 += 2') == ('self', '.', 'x0', ' += 2')
    '''

    tokens = []
    is_in_id = False

    for c in s:
        if c.isidentifier() or (is_in_id and c.isdigit()):
            if not is_in_id:
                tokens.append('')
                is_in_id = True
        else:
            if is_in_id or len(tokens) == 0:
                tokens.append('')
                is_in_id = False
        tokens[-1] += c

    return tuple(tokens)

def extract_identifiers(l):
    tokens = split_at_identifier_boundaries(l)
    return [t for t in tokens if t[0].isidentifier()]

def build_github_dataset(output, language = 'Python'):
    def load_github(path):
        with open(path) as f:
            f = json.load(f)
            return (f[language]['train'], f[language]['dev'], f[language]['test'])

    train, val, test = (load_github('datasets/files-python.json'))

    def make_examples(files):
        ex = []
        for f in tqdm(files):
            lines = list(filter(None, [l.strip() for l in f.split('\n')]))

            for i, l in enumerate(lines):
                if i < 2:
                    continue
                identifiers = extract_identifiers(l)

                if len(identifiers) < 3:
                    continue

                context = lines[i-2:i]
                signal = ' '.join(random.sample(identifiers, random.randint(1, 3)))
                answer = l
                ex.append(Example(context, signal, answer))
        return ex

    print(len(train) + len(val) + len(test), language, 'files loaded.')
    tokenizer = train_tokenizer(train)
    random.seed('cogents-code')
    train_ex = make_examples(train)
    val_ex = make_examples(val)
    test_ex = make_examples(test)
    breakpoint()
    d = Dataset(train_ex, val_ex, test_ex, tokenizer)
    print(len(train_ex), 'training examples.')
    torch.save(d, output)
