#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import OpenAIGPTConfig, OpenAIGPTModel
from tokenizers import Tokenizer

from data import Example, Dataset


class CogentModel(pl.LightningModule):
    def forward(self, batch):
        pass

    def generate(self, batch):
        pass


class VanillaTransformer(pl.LightningModule):
    def __init__(self, tokenizer=None, config={}, device=None):
        super().__init__()

        embedding_size = config.get('embedding_dim', 256)

        self.gpt_config = OpenAIGPTConfig(
            vocab_size=tokenizer.get_vocab_size(),
            n_positions=256,
            n_embd=embedding_size,
            n_head=4,
            n_layer=3,
        )
        self.gpt = OpenAIGPTModel(self.gpt_config)
        self.embeddings = nn.Embedding(tokenizer.get_vocab_size(),
                                       embedding_size)
        self.output = nn.Linear(embedding_size, tokenizer.get_vocab_size())
        self.tokenizer = tokenizer

    def forward(self, batch):
        pass

    def generate(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        x_encs = self.tokenizer.encode_batch([e.context + "[SEP]" + e.answer for e in batch])
        max_len = max([len(e.ids) for e in x_encs])

        for e in x_encs:
            e.pad(max_len)

        x_emb = self.embeddings(torch.tensor([e.ids for e in x_encs], device=self.device))
        y = self.output(self.gpt(inputs_embeds=x_emb).last_hidden_state)
        y = y.transpose(1, 2)
        y = y[:, :, :-1]

        loss = nn.CrossEntropyLoss()

        target = torch.tensor([e.ids for e in x_encs], device=self.device) # (B, S)
        target = target[:, 1:]
        sep_index = (target == self.tokenizer.token_to_id('[SEP]'))
        after_sep = sep_index.cumsum(dim=1).long()
        after_sep -= sep_index.long()
        target *= after_sep
        return loss(y, target)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class SignalTransformer(VanillaTransformer):
    def training_step(self, batch, batch_idx):
        for e in batch:
            e.context = e.signal + '[CLS]' + e.context
        return(VanillaTransformer.training_step(self, batch, batch_idx))




def train_model(dataset_path, devices, transformer):
    dataset = torch.load(dataset_path)
    print('Loaded dataset', dataset_path)

    trainer = pl.Trainer(devices=devices, accelerator="auto")
    train_loader = DataLoader(dataset.train, batch_size=32, collate_fn=list)

    if transformer=='vanilla':
        model = VanillaTransformer(dataset.tokenizer)
    elif transformer=='signal':
        model = SignalTransformer(dataset.tokenizer)

    trainer.fit(model, train_loader)
