#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from torch.distributions.categorical import Categorical
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model
from tokenizers import Tokenizer
from pytorch_lightning.loggers import WandbLogger

from data import Example, Dataset


class CogentModel(pl.LightningModule):
    def forward(self, batch):
        pass

    def generate(self, batch):
        pass


class VanillaTransformer(pl.LightningModule):
    def __init__(self, tokenizer=None, config={}, device=None):
        super().__init__()

        # embedding_size = config.get('embedding_dim', 512)
        embedding_size = config.get('embedding_dim', 768)

        self.gpt_config = GPT2Config(
            vocab_size=tokenizer.get_vocab_size(),
            n_positions=512,
            n_ctx=512,
            n_embd=embedding_size,
            n_head=12,
            n_layer=12,
            pad_token_id=tokenizer.token_to_id('[PAD]'),
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
        emb_index = torch.arange(max_len).repeat((len(batch), 1)).to(device=self.device)
        mask = torch.ones_like(emb_index)

        for i, e in enumerate(x_encs):
            emb_index[i, len(e):] = len(e) - 1
            mask[i, len(e):] = 0
            e.pad(max_len)

        x_emb = self.embeddings(torch.tensor([e.ids for e in x_encs], device=self.device))
        return x_encs, x_emb, mask, emb_index

    def forward(self, x_emb, mask, emb_index, past_kv=None):
        gpt_output = self.gpt(inputs_embeds=x_emb,
                              attention_mask=mask,
                              position_ids=emb_index,
                              past_key_values=past_kv,
                              use_cache=True)
        y = self.output(gpt_output.last_hidden_state)
        return y, gpt_output.past_key_values

    def generate(self, batch, max_len=50):
        x_encs, x_emb, mask, emb_index = self.encode_batch(batch)
        past_kv = None

        done = set()
        outputs = [list() for _ in batch]
        pred_index = emb_index.max(axis=1).values

        with torch.no_grad():
            j = 0
            while len(done) < len(batch) and j < max_len:
                j += 1
                y, past_kv = self.forward(x_emb, mask, emb_index, past_kv)
                y = y[torch.arange(len(batch)), pred_index, :]
                # Increment position embedding index.
                emb_index = emb_index.max(axis=1, keepdim=True).values + 1
                # Use all tokens after first iteration.
                next_y = Categorical(logits=y).sample()
                mask = torch.ones((y.shape[0], 1), device=self.device)
                x_emb = self.embeddings(next_y.unsqueeze(1))
                pred_index = torch.zeros(y.shape[0], dtype=torch.long, device=self.device)

                for i, t in enumerate(next_y):
                    if t == self.tokenizer.token_to_id('[EOS]'):
                        done.add(i)
                    if i not in done:
                        outputs[i].append(t)

        return [self.tokenizer.decode(o) for o in outputs]

    def training_step(self, batch, batch_idx, log=True, restrict_loss=True):
        x_encs, x_emb, mask, emb_index = self.encode_batch(batch, append_answer=True)
        y, last_kv = self.forward(x_emb, mask, emb_index)

        # For training, ignore prediction on last token ([EOS]).
        y = y.transpose(1, 2)[:, :, :-1]

        # By default, celoss ignores columns where target label is -100.
        celoss = nn.CrossEntropyLoss()

        target = torch.tensor([e.ids for e in x_encs], device=self.device) # (B, S)
        target = target[:, 1:]
        sep_index = (target == self.tokenizer.token_to_id('[SEP]'))
        after_sep = sep_index.cumsum(dim=1).long()
        after_sep -= sep_index.long()

        eos_index = (target == self.tokenizer.token_to_id('[EOS]'))
        after_eos = eos_index.cumsum(dim=1).long()
        after_eos -= eos_index.long()

        pred_range = after_sep - after_eos

        # For computing the loss, overwrite all tokens before [SEP] as -100 tokens
        # since those are ignored by celoss.
        target = pred_range * target + (1 - pred_range) * -100
        # breakpoint()
        loss = celoss(y, target)

        if log:
            self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log=False)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer


class SignalTransformer(VanillaTransformer):
    def training_step(self, batch, batch_idx, *args, **kwargs):
        batch = [Example(e.context + '[SIG]' + e.signal, e.signal, e.answer) for e in batch]
        return VanillaTransformer.training_step(self, batch, batch_idx, *args, **kwargs)

    def generate(self, batch, *args, **kwargs):
        batch = [Example(e.context + '[SIG]' + e.signal, e.signal, e.answer) for e in batch]
        return VanillaTransformer.generate(self, batch, *args, **kwargs)

def train_model(dataset_path, devices, transformer, output_path):
    dataset = torch.load(dataset_path)
    print('Loaded dataset', dataset_path)
    print('Using devices', devices)

    logger = WandbLogger(project="cogent")

    trainer = pl.Trainer(devices=devices, accelerator="auto", strategy='ddp', logger=logger)

    train_loader = DataLoader(dataset.train, batch_size=32, collate_fn=list, shuffle=True)
    val_loader = DataLoader(dataset.val, batch_size=32, collate_fn=list)

    if transformer == 'vanilla':
        model = VanillaTransformer(dataset.tokenizer)
    elif transformer == 'signal':
        model = SignalTransformer(dataset.tokenizer)

    trainer.fit(model, train_loader, val_loader)
    torch.save(model, output_path)


def generate_from_model(dataset_path, model_path, transformer, device):
    dataset = torch.load(dataset_path)
    print('Loaded dataset', dataset_path)

    if model_path is None:
        model = VanillaTransformer(dataset.tokenizer)
        model.to(device=device)
    else:
        pl_state = torch.load(model_path, map_location=device)

        if transformer == 'vanilla':
            model = VanillaTransformer(dataset.tokenizer)
        elif transformer == 'signal':
            model = SignalTransformer(dataset.tokenizer)

        model.load_state_dict(pl_state['state_dict'])
        model.to(device)

    test_loader = DataLoader(dataset.test, batch_size=32, collate_fn=list, shuffle=False)

    for batch in test_loader:
        g = model.generate(batch)

        for e, pred in zip(batch, g):
            print('#' * 50)
            print('Context:', e.context)
            print('Signal:', e.signal)
            print('Ground truth:', e.answer)
            print('Sample from model:', pred)
