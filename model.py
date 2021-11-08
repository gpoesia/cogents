#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from torch.distributions.categorical import Categorical
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model
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

        self.gpt_config = GPT2Config(
            vocab_size=tokenizer.get_vocab_size(),
            n_positions=512,
            n_embd=embedding_size,
            n_head=4,
            n_layer=3,
        )
        self.gpt = GPT2Model(self.gpt_config)
        self.embeddings = nn.Embedding(tokenizer.get_vocab_size(),
                                       embedding_size)
        self.output = nn.Linear(embedding_size, tokenizer.get_vocab_size())
        self.tokenizer = tokenizer

    def encode_batch(self, batch, append_answer=False):
        x_encs = self.tokenizer.encode_batch(["[BOS]" +
                                              e.context +
                                              "[SEP]" +
                                              (e.answer + "[EOS]" if append_answer else "")
                                            for e in batch])
        max_len = max([len(e.ids) for e in x_encs])

        for e in x_encs:
            e.pad(max_len)

        x_emb = self.embeddings(torch.tensor([e.ids for e in x_encs], device=self.device))
        return x_encs, x_emb

    def forward(self, x_emb, past_kv=None):
        gpt_output = self.gpt(inputs_embeds=x_emb, past_key_values=past_kv, use_cache=True)
        y = self.output(gpt_output.last_hidden_state)
        return y, gpt_output.past_key_values

    def generate(self, batch, max_len=50):
        x_encs, x_emb = self.encode_batch(batch)
        past_kv = None

        done = set()
        outputs = [list() for _ in batch]

        with torch.no_grad():
            j = 0
            while len(done) < len(batch) and j < max_len:
                j += 1
                y, past_kv = self.forward(x_emb, past_kv)
                y = y[:, -1, :]
                next_y = Categorical(logits=y).sample()
                x_emb = self.embeddings(next_y.unsqueeze(1))

                for i, t in enumerate(next_y):
                    if t == self.tokenizer.token_to_id('[EOS]'):
                        done.add(i)
                    if i not in done:
                        outputs[i].append(t)

        return [self.tokenizer.decode(o) for o in outputs]

    def training_step(self, batch, batch_idx):
        x_encs, x_emb = self.encode_batch(batch, append_answer=True)
        y, last_kv = self.forward(x_emb)

        # For training, ignore prediction on last token ([EOS]).
        y = y.transpose(1, 2)[:, :, :-1]

        loss = nn.CrossEntropyLoss()

        target = torch.tensor([e.ids for e in x_encs], device=self.device) # (B, S)
        target = target[:, 1:]
        sep_index = (target == self.tokenizer.token_to_id('[SEP]'))
        after_sep = sep_index.cumsum(dim=1).long()
        after_sep -= sep_index.long()
        target *= after_sep
        return loss(y, target)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class SignalTransformer(VanillaTransformer):
    def training_step(self, batch, batch_idx):
        for e in batch:
            e.context = e.signal + '[CLS]' + e.context
        return(VanillaTransformer.training_step(self, batch, batch_idx))

def train_model(dataset_path, devices, transformer, output_path):
    dataset = torch.load(dataset_path)
    print('Loaded dataset', dataset_path)

    trainer = pl.Trainer(devices=devices, accelerator="auto")
    train_loader = DataLoader(dataset.train, batch_size=128, collate_fn=list)
    val_loader = DataLoader(dataset.val, batch_size=64, collate_fn=list)

    if transformer == 'vanilla':
        model = VanillaTransformer(dataset.tokenizer)
    elif transformer == 'signal':
        model = SignalTransformer(dataset.tokenizer)

    trainer.fit(model, train_loader, val_loader)
    torch.save(model, output_path)


def generate_from_model(dataset_path, model_path, device):
    dataset = torch.load(dataset_path)
    print('Loaded dataset', dataset_path)

    if model_path is None:
        model = VanillaTransformer(dataset.tokenizer)
        model.to(device=device)
    else:
        pl_state = torch.load(model_path, map_location=torch.device('cpu'))
        model = VanillaTransformer(dataset.tokenizer)
        model.load_state_dict(pl_state['state_dict'])
        breakpoint()

    test_loader = DataLoader(dataset.train, batch_size=64, collate_fn=list)

    for batch in test_loader:
        g = model.generate(batch)
        break

    print(g)
