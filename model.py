#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from torch.distributions.categorical import Categorical
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model, BertModel, BertConfig
from tokenizers import Tokenizer
from pytorch_lightning.loggers import WandbLogger

from data import Example, Dataset
import utils


class CogentModel(pl.LightningModule):
    def forward(self, batch):
        pass

    def generate(self, batch):
        pass


class VanillaTransformer(pl.LightningModule):
    def __init__(self, tokenizer=None, config={}, device=None, n_head = 12, n_layer=12):
        super().__init__()

        # embedding_size = config.get('embedding_dim', 512)
        embedding_size = config.get('embedding_dim', 768)

        self.gpt_config = GPT2Config(
            vocab_size=tokenizer.get_vocab_size(),
            n_positions=512,
            n_ctx=512,
            n_embd=embedding_size,
            n_head=n_head,
            n_layer=n_layer,
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

    def training_step(self, batch, batch_idx, log=True):
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


class CogentTransformer(VanillaTransformer):
    def __init__(self, tokenizer=None, config={}, device=None, n_head = 2, n_layer=2):
        super().__init__(tokenizer, config, device, n_head, n_layer)

        embedding_dim = config.get('embedding_dim', 768)

        bert_config = BertConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=embedding_dim,
            num_hidden_layers=3,
            num_attention_heads=8,
            intermediate_size=4*embedding_dim,
            max_position_embeddings=512,
        )
        self.lr = 5e-5

        self.bert = BertModel(bert_config)
        # Turn embedding into mean/variance parameters
        self.inference_proj = nn.Linear(embedding_dim,
                                        2*embedding_dim)
        self.z_prior_mu = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_var = nn.Parameter(torch.ones(1), requires_grad=False)

    def z_inference_forward(self, batch):
        x_encs, x_emb, mask, emb_index = self.encode_target(batch)
        bert_out = self.bert(inputs_embeds=x_emb, attention_mask=mask)
        embeddings = bert_out.last_hidden_state[:, 0, :]
        y_hat = self.inference_proj(embeddings)
        mean, variance = utils.gaussian_parameters(y_hat, dim=1)
        return mean, variance

    def encode_input(self, batch, append_answer=True):
        x_encs = self.tokenizer.encode_batch(["[BOS]" +
                                              e.context +
                                              '[SIG]' +
                                              e.signal +
                                              "[SEP]" +
                                              (e.answer + "[EOS]" if append_answer else "")
                                              for e in batch])
        return x_encs, *self.embed_batch(x_encs)

    def encode_target(self, batch):
        x_encs = self.tokenizer.encode_batch(["[BOS]" +
                                              e.answer +
                                              "[EOS]"
                                              for e in batch])
        return x_encs, *self.embed_batch(x_encs)

    def embed_batch(self, x_encs):
        max_len = max([len(e.ids) for e in x_encs])
        emb_index = torch.arange(max_len).repeat((len(x_encs), 1)).to(device=self.device)
        mask = torch.ones_like(emb_index)

        for i, e in enumerate(x_encs):
            emb_index[i, len(e):] = len(e) - 1
            mask[i, len(e):] = 0
            e.pad(max_len)

        x_emb = self.embeddings(torch.tensor([e.ids for e in x_encs], device=self.device))
        return x_emb, mask, emb_index

    def forward(self, x_emb, mask, emb_index=None, past_kv=None):
        gpt_output = self.gpt(inputs_embeds=x_emb,
                              attention_mask=mask,
                              position_ids=emb_index,
                              past_key_values=past_kv,
                              use_cache=True)
        # Ignore 'z' token in getting y.
        y = self.output(gpt_output.last_hidden_state[:, (1 if past_kv is None else 0):, :])
        return y, gpt_output.past_key_values

    def generate(self, batch, max_len=50):
        x_encs, x_emb, mask, emb_index = self.encode_input(batch, append_answer=False)
        z = utils.sample_gaussian(self.z_prior_mu.repeat((x_emb.shape[0], 1, x_emb.shape[2])),
                                  self.z_prior_var.repeat((x_emb.shape[0], 1, x_emb.shape[2])))

        # Append a "z" prefix token before the other tokens.
        x_emb = torch.cat([z, x_emb], dim=1)
        mask = torch.cat([torch.ones((mask.shape[0], 1), device=self.device), mask], dim=1)

        emb_index[emb_index == -1] = -2
        emb_index = torch.cat([torch.zeros((mask.shape[0], 1), device=self.device, dtype=torch.long),
                               emb_index + 1], dim=1)

        past_kv = None

        done = set()
        outputs = [[] for _ in batch]
        pred_index = emb_index.max(axis=1).values - 1

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

    def training_step(self, batch, batch_idx, log=True):
        x_encs, x_emb, mask, emb_index = self.encode_input(batch, append_answer=True)
        z_mu, z_var = self.z_inference_forward(batch)

        z = utils.sample_gaussian(z_mu, z_var)

        # Append a "z" prefix token before the other tokens.
        x_emb = torch.cat([z.unsqueeze(1), x_emb], dim=1)
        mask = torch.cat([torch.ones((mask.shape[0], 1), device=self.device), mask], dim=1)

        y, last_kv = self.forward(x_emb, mask)

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
        rec_loss = celoss(y, target)
        kl_loss = utils.kl_normal(z_mu, z_var, self.z_prior_mu, self.z_prior_var).mean(dim=0)
        nelbo = kl_loss + rec_loss
        loss = nelbo

        if log:
            self.log('train_loss', loss)
            self.log('rec_loss', rec_loss)
            self.log('kl_loss', kl_loss)

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


def train_model(dataset_path, devices, transformer, output_path, strat='ddp', n_head = 12, n_layer = 12):
    dataset = torch.load(dataset_path)
    print('Loaded dataset', dataset_path)
    print('Using devices', devices)

    logger = WandbLogger(project="cogent")

    train_loader = DataLoader(dataset.train, batch_size=32, collate_fn=list, shuffle=True)
    val_loader = DataLoader(dataset.val, batch_size=32, collate_fn=list)

    if transformer == 'vanilla':
        model = VanillaTransformer(dataset.tokenizer, n_head=n_head, n_layer=n_layer)
    elif transformer == 'signal':
        model = SignalTransformer(dataset.tokenizer)
    elif transformer == 'cogent':
        model = CogentTransformer(dataset.tokenizer, n_head=n_head, n_layer=n_layer)

    ckpt = pl.callbacks.ModelCheckpoint(dirpath=f'checkpoints/',
                                        monitor='validation_loss',
                                        save_top_k=-1,
                                        filename='{transformer}-{epoch}')

    if strat =='ddp':
        trainer = pl.Trainer(devices=devices, accelerator="auto", strategy=strat, logger=logger, max_epochs=10, checkpoint_callback=ckpt)
    else:
        trainer = pl.Trainer(devices=devices, accelerator="auto", logger=logger)

    trainer.fit(model, train_loader, val_loader)
    torch.save(model, output_path)


def generate_from_model(dataset_path, model_path, transformer, device, eval_perplexity):
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

    perplexities = []
    for batch in test_loader:
        g = model.generate(batch)
        for e, pred in zip(batch, g):
            print('#' * 50)
            print('Context:', e.context)
            print('Signal:', e.signal)
            print('Ground truth:', e.answer)
            print('Sample from model:', pred)
            if eval_perplexity:
                loss = model.training_step(batch, 0, log=False).detach()
                perplexity = torch.exp(loss)
                perplexities.append(perplexity)
                print('Perplexity:', perplexity)
        if eval_perplexity:
            perplexities = torch.tensor(perplexities)
            avg_perplexity = torch.mean(perplexities)
            print('Average Perplexity:', avg_perplexity)
            return avg_perplexity
