# =============================================================================
# Libs
# =============================================================================
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re

import os
import time
import argparse
import torch.distributed as dist
from fairscale.optim.oss import OSS
from fairscale.utils.testing import set_random_seed
# import functools
# from torch.nn.parallel import DistributedDataParallel as DDP
# from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
# from fairscale.nn.wrap import enable_wrap, auto_wrap, default_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
# # from torchinfo import summary
from fairscale.nn.model_parallel import initialize_model_parallel, destroy_model_parallel
# import traceback
# # dudu additions
# from dudu_tests import custom_wrap
# from dudu_tests.layerwise_data_parallel import DataParallelLayer
# from dudu_tests.layerwise_model_parallel import ColumnParallelLinear, RowParallelLinear
from dudu_tests import strategy_handler
import json
import dudu_tests.Optimizer as Optimizer
from dudu_tests.layerwise_data_parallel import _gather
from datetime import timedelta


# =============================================================================
# Transformer
# =============================================================================
def attention(q, k, v, mask=None, dropout=None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])

    # mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output


# # todo figure out what to do with this layer, should i only parallelize layers with parameters? for now, yes
# class AddLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)
#
#     def forward(self, x, y):
#         return x + y


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        """
        n_heads: the number of attention heads
        out_dim: embedding size
        dropout: chance of dropout for inner linear layers
        """
        super().__init__()

        #        self.q_linear = nn.Linear(out_dim, out_dim)
        #        self.k_linear = nn.Linear(out_dim, out_dim)
        #        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim * 3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def forward(self, x, y=None, mask=None):
        # in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y

        qkv = self.linear(x)  # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim]  # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim * 2]  # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim * 2:]  # BS * SEQ_LEN * EMBED_SIZE_L

        # break into n_heads
        q, k, v = [self.split_heads(t) for t in (q, k, v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD

        # n_heads => attention => merge the heads => mix information
        scores = attention(q, k, v, mask, self.dropout)  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1, 2).contiguous().view(scores.shape[0], -1,
                                                          self.out_dim)  # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE

        return out


class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # # todo delete this if fails. regardless, find a better solution to this
        # self.add1 = AddLayer()
        # self.add2 = AddLayer()

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        # # todo delete this if fails. regardless, find a better solution to this
        # x = self.add1(x, self.dropout1(self.mha(x2, mask=mask)))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        # # todo delete this if fails. regardless, find a better solution to this
        # x = self.add2(x, self.dropout2(self.ff(x2)))
        return x


class Transformer(nn.Module):
# class Transformerz(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, dropout=.1):
        super().__init__()
        assert embed_size % n_heads == 0
        # model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        # to turn gradients off
        # self.embeddings.weight.requires_grad = False
        self.pe = PositionalEmbedding(embed_size, seq_len)
        # # todo delete this if fails. regardless, find a better solution to this
        # self.add = AddLayer()

        # backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)

        # self.encoders = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_size,
        #                                                                  dim_feedforward=inner_ff_size, nhead=n_heads,
        #                                                                  batch_first=True, dropout=dropout),
        #                                       num_layers=n_code, norm=nn.LayerNorm(embed_size))

        # language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, n_embeddings, bias=False)

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.pe(x)
        # # todo delete this if fails. regardless, find a better solution to this
        # x = self.add(x, self.pe(x))
        # x = self.encoders(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        # return x
        return x.view(-1, x.shape[-1])


# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # x.size(1) = seq_len


# =============================================================================
# Dataset
# =============================================================================
class SentencesDataset(Dataset):
    # Init dataset
    def __init__(self, sentences, vocab, seq_len):
        dataset = self

        dataset.sentences = sentences
        dataset.vocab = vocab + ['<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e: i for i, e in enumerate(dataset.vocab)}
        dataset.rvocab = {v: k for k, v in dataset.vocab.items()}
        dataset.seq_len = seq_len

        # special tags
        dataset.IGNORE_IDX = dataset.vocab['<ignore>']  # replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>']  # replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>']  # replacement tag for the masked word prediction task

    # fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self

        # while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1

        # ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))]  # PAD ok

        # apply random mask
        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask else (w, dataset.IGNORE_IDX) for w in s]

        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}

    # return length
    def __len__(self):
        return len(self.sentences)

    # get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s]
        return s


# =============================================================================
# Methods / Class
# =============================================================================
def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def init_processes(train_args):
    # These are the parameters used to initialize the process group
    local_rank = int(os.environ['LOCAL_RANK'])
    # currently its assumed that it will use the default initialization method to sync all workers across nodes
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=1.5))
    # initialize model parallel process: split layer across 1 machines and pipeline length is 1
    # data parallel size = world_size / (model * pipeline parallel)
    backends = {"model_parallel_backend": "nccl", "pipeline_backend": "nccl", "ddp_backend": "nccl"}
    initialize_model_parallel(dist.get_world_size(), 1, **backends)
    set_random_seed(int(time.strftime("%M")))
    main_loop(local_rank, train_args)
    destroy_model_parallel()
    dist.destroy_process_group()


def main_loop(rank, train_args):
    # =============================================================================
    # #Init
    # =============================================================================
    global_rank = dist.get_rank()
    print(f"[{os.getpid()}] rank = {global_rank}, " + f"world_size = {dist.get_world_size()}")
    torch.cuda.set_device(rank)

    # if train_args.import_strategy:
    #     train_args.strategy = strategy_handler.import_strategy(train_args.import_strategy)
    # # if no valid strategy is given, create a new one using the wrap feature
    # else:
    #     # todo delete this and swap with vanilla strategy creator once implemented in optimizer
    #     train_args.strategy = strategy_handler.import_strategy('strategies/bert_vanilla.json')
    train_args.n_iteration = 10
    avg_ips, latency = train_and_test(rank, train_args)
    if rank == 0:
        print(f'Best Latency: {latency}, Best strategy:\n{train_args.strategy} ')


# TODO unpack into a list of understandable arguments and pass train_args as **train_args
def train_and_test(rank, train_args):
    seq_len = train_args.seq_length
    embed_size = train_args.embed_size
    inner_ff_size = train_args.hidden_layer_size
    n_heads = train_args.atten_heads
    n_code = train_args.encoder_layers
    n_vocab = 40000
    dropout = 0.1
    n_workers = 12

    # optimizer
    optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4, 'betas': (.9, .999)}

    # =============================================================================
    # Input
    # =============================================================================
    # 1) load text
    # print(f'loading text in global rank {global_rank}')
    pth = 'europarl30k.fr.txt'
    with open(pth, mode='r') as file:
        sentences = file.read().lower().split('\n')

    # 2) tokenize sentences (can be done during training, you can also use spacy udpipe)
    # print(f'tokenizing sentences in global rank {global_rank}')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]

    # 3) create vocab if not already created
    # print(f'creating/loading vocab in global rank {global_rank}')
    pth = 'vocab.txt'
    if not exists(pth):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(n_vocab)  # keep the N most frequent words
        vocab = [w[0] for w in vocab]
        with open(pth, 'w+') as file:
            file.write('\n'.join(vocab))
    else:
        with open(pth, mode='r') as file:
            vocab = file.read().split('\n')

    # 4) create dataset
    # print(f'creating dataset in global rank {global_rank}')
    dataset = SentencesDataset(sentences, vocab, seq_len)
    # sampler = None
    sampler = DistributedSampler(dataset)
    kwargs = {'num_workers': n_workers, 'shuffle': False, 'drop_last': True, 'pin_memory': True,
              'batch_size': train_args.batch_size, 'sampler': sampler}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    # =============================================================================
    # Find Optimal Strategy
    # =============================================================================
    # wrap the layers of the model with data parallel layers that will enable custom data parallelism
    if rank == 0:
        print(
            f'seq len: {seq_len}, embed size: {embed_size}, inner ff size: {inner_ff_size}, n heads: {n_heads},'
            f' encoder layers: {n_code}')
    print(f'finding optimal strategy')
    loss_model = nn.CrossEntropyLoss()
    # for running without oss
    temp_model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.vocab), seq_len, dropout)
    temp_optimizer = optim.Adam(temp_model.parameters(), **optim_kwargs)
    sample = next(iter(data_loader))
    # infer
    masked_input = sample['input']
    masked_target = sample['target']
    masked_input = masked_input.to(rank)
    masked_target = masked_target.to(rank)
    # todo delete and find a more elegant way to do this
    # targets = _gather(masked_target, batch=True)
    # targets = targets.view(-1, 1).squeeze()
    targets = masked_target.view(-1, 1).squeeze()
    train_args.strategy, best_latency = Optimizer.find_optimal_strategy(temp_model, temp_optimizer, loss_model,
                                                                        masked_input, targets, train_args.budget,
                                                                        alpha=0.005,
                                                                        sim_iterations=train_args.n_simulations,
                                                                        imported_strat=train_args.import_strategy)
    if rank == 0 and train_args.print_params:
        print(f'optimal latency: {best_latency}, optimal strategy:\n{train_args.strategy}')
    if train_args.export_strategy:
        strategy_handler.export_strategy(strategy=train_args.strategy, target_location=train_args.export_strategy)
    # =============================================================================
    # Optimizer
    # =============================================================================
    print(f'initializing optimizer and loss  in global rank {dist.get_rank()}')
    loss_model = nn.CrossEntropyLoss()
    model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.vocab), seq_len, dropout)
    model = Optimizer.wrap_model(model, train_args.strategy)
    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    # =============================================================================
    # Train
    # =============================================================================
    print_each = 10
    model.train()
    start = time.time()
    n_iteration = train_args.n_iteration
    for e in range(train_args.epochs):
        batch_iter = iter(data_loader)
        for it in range(n_iteration):
        # for it, batch in enumerate(data_loader, 0):
            # get batch
            batch, batch_iter = get_batch(data_loader, batch_iter)
            # infer
            masked_input = batch['input']
            masked_target = batch['target']

            masked_input = masked_input.to(rank)
            masked_target = masked_target.to(rank)

            output_v = model(masked_input)
            # compute the cross entropy loss
            target_v = masked_target.view(-1, 1).squeeze()
            # todo delete and find a more elegant way to do this
            target_v = _gather(target_v, batch=True)
            loss = loss_model(output_v, target_v)
            # compute gradients
            loss.backward()
            # extra step to make sure all gradients are in sync
            Optimizer.sync_gradients(model)
            # apply gradients
            optimizer.step()

            if it >= n_iteration:
                break
            # print step
            if it % print_each == 0 and rank == 0:
                # print('it:', it,
                #       ' | loss', np.round(loss.item(), 2),
                #       ' | Δw:', round(model.embeddings.weight.grad.abs().sum().item(), 3))
                print('it:', it,
                      ' | loss', np.round(loss.item(), 2))
            # reset gradients
            optimizer.zero_grad()
    tot_time = time.time() - start
    ips = (train_args.epochs * n_iteration * train_args.batch_size * dist.get_world_size()) / tot_time
    if rank == 0 and train_args.print_params:
        print(f"GPU AMOUNT = {dist.get_world_size()}, ELAPSED TIME = {tot_time}s, THROUGHPUT = {ips} samples/s, "
              # f"latency = {tot_time / (train_args.epochs * n_iteration)}")
              f"avg_latency = {tot_time / (train_args.epochs)}, BATCH: {train_args.batch_size}")
    return ips, tot_time / (train_args.epochs * n_iteration)


def recurse_print_layers(module: torch.nn.Module, base_name=''):
    print_str = f'name: {base_name}, type: {type(module)}'
    if isinstance(module, MultiHeadAttention):
        return print_str
    print_str += '\n    '
    for name, layer in module.named_children():
        print_str += recurse_print_layers(layer, name)
    return print_str


if __name__ == "__main__":
    """
    a good source to decide the size of the word embeddings is:
    https: // aclanthology.org / I17 - 2006 /
    """
    parser = argparse.ArgumentParser()
    # # ======================== LARGE NET TEST (more than 12GB parameters)
    # parser.add_argument("--seq_length", type=int, default=20)
    # parser.add_argument("--embed_size", type=int, default=240)
    # parser.add_argument("--hidden_layer_size", type=int, default=(240 * 4))
    # parser.add_argument("--atten_heads", type=int, default=16)
    # parser.add_argument("--encoder_layers", type=int, default=24)
    # parser.add_argument("--batch_size", type=int, default=512)

    # ======================= BASE NET TEST
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--embed_size", type=int, default=240)
    parser.add_argument("--hidden_layer_size", type=int, default=(240 * 4))
    parser.add_argument("--atten_heads", type=int, default=12)
    # parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)

    # # ======================= SUPER LARGE NET TEST
    # parser.add_argument("--seq_length", type=int, default=20)
    # parser.add_argument("--embed_size", type=int, default=240)
    # parser.add_argument("--hidden_layer_size", type=int, default=(240 * 4))
    # parser.add_argument("--atten_heads", type=int, default=16)
    # parser.add_argument("--encoder_layers", type=int, default=50)
    # parser.add_argument("--batch_size", type=int, default=512)

    # parser.add_argument('--print_params', dest='print_params', default=False, action='store_true')
    parser.add_argument("--n_simulations", type=int, default=8)
    parser.add_argument('--print_params', dest='print_params', default=False, action='store_true')
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--budget", action="store", default=10, type=int)
    parser.add_argument("--import_strategy", action="store", default='', type=str)
    parser.add_argument("--export_strategy", action="store", default='strategies/exported_strategy.json', type=str)
    parser.add_argument("--name", action="store", default='no_name.pt', type=str)
    args = parser.parse_args()

    init_processes(args)


