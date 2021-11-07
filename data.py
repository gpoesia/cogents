#!/usr/bin/env python3

import csv
from dataclasses import dataclass
from typing import List, Dict
import random

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.trainers import BpeTrainer


@dataclass
class Example:
    context: List[int]
    signal: List[int]
    answer: List[int]


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
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoders = decoders.ByteLevel()

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
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


def build_rocstories_dataset():
    def load_rocstories(path):
        with open(path) as f:
            return [ShortStory(title=row['storytitle'],
                               lines=[row[f'sentence{i}'] for i in range(1, 6)])
                    for row in csv.DictReader(f)]

    rocstories = (load_rocstories('datasets/rocstories_2016.csv') +
                  load_rocstories('datasets/rocstories_2017.csv'))

    def make_examples(stories):
        ex = []

        for s in stories:
            for i in range(10):
                for k in range(1, 5):
                    target_line = random.randint(0, len(s.lines) - 1)
                    context_lines = [s.title] + s.lines[:target_line]
                    target_words = s.lines[target_line].split()
                    signal = random.sample(target_words, min(k, len(target_words)))
                    ex.append(Example('\n'.join(context_lines), ' '.join(signal), s.lines[target_line]))

        return ex

    print(len(rocstories), 'stories loaded.')

    random.seed('cogents-rocstories')
    (train, val, test) = random_split(rocstories, [0.8, 0.1, 0.1])
    tokenizer_data = [s.title for s in train] + [l for s in train for l in s.lines]
    tokenizer = train_tokenizer(tokenizer_data)

    train_ex, val_ex, test_ex = make_examples(train), make_examples(val), make_examples(test)
    return Dataset(train_ex, val_ex, test_ex, tokenizer)
