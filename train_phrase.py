# Training this with multi-keyword from each sentence

import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from utils.tf_adpt import GPT2LMHeadModel
from tqdm import tqdm
import random
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from clip1 import clip
from clip1.clip import _transform
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import skimage.io as io1
from PIL import Image
from PIL import ImageFile
from timm.models.layers import trunc_normal_
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from project.scheduler.lr_scheduler import build_scheduler
from utils.misc import generate2_adpt_if, evaluate_on_coco_caption
import torch.nn.functional as F
import nltk
from project.models.model import MappingType, ClipCaptionModel, ClipCaptionPrefix, index_to_log_onehot, log_onehot_to_index


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        gt = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        padding = self.max_seq_len - tokens.shape[0]
        pad_gt = self.max_seq_len - gt.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        if pad_gt > 0:
            gt = torch.cat((gt, torch.zeros(pad_gt, dtype=torch.int64) - 1))  # we set target == -1 as ignore target
        elif pad_gt < 0:
            gt = gt[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        return tokens, mask, gt

    def key_tokens(self, item: int):
        cap_stc = self.captions[item]
        is_noun = lambda pos: pos[:2] == 'NN'
        is_verb = lambda pos: pos[:2] == 'VB'
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', 'A', 'The', 'To', 'An']
        stop_words = nltk.corpus.stopwords.words("english")
        cap_words = nltk.word_tokenize(cap_stc)
        cap_kws = [word for (word, pos) in nltk.pos_tag(cap_words) if ((word not in english_punctuations) and (word not in stop_words) and (is_noun or is_verb))]
        if not cap_kws:
            cap_kws.append(' ')
        kw_list = random.sample(cap_kws, random.choice(range(1, len(cap_kws))))
        kw = ''
        for key in kw_list:
            kw = kw + key + ','
        kt = torch.tensor(self.tokenizer.encode(kw))

        padding = self.max_key_len - kt.shape[0]
        if padding > 0:
            kt = torch.cat((kt, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            kt = kt[:self.max_key_len]
        mask_kt = kt.ge(0)
        kt[~mask_kt] = 0
        mask_kt = mask_kt.float()
        return kt, mask_kt
    
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, gt = self.pad_tokens(item)
        img_id = self.image_ids[item]
        kt, mask_kt = self.key_tokens(item)
        # mask = torch.cat((mask_kt, mask), dim=0)
        # train+restval
        filename = f"{self.data_root}/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        try:
            image = io1.imread(filename)
        except:
            filename = f"{self.data_root}/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        ## this is for pre-computed clip feature
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, image, gt, kt

    def __init__(self, data_root: str, data_path: str, gpt2_type: str = "gpt2", len_seq=20, len_key=20, normalize_prefix=False):
        self.data_root = data_root
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.normalize_prefix = normalize_prefix
        self.preprocess = _transform(224)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])  # just index
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        ## notice current one is 40
        # self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        self.max_seq_len = len_seq
        self.max_key_len = len_key


class ClipCocoValDataset(Dataset):

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int):
        _filename = self.files[item]
        filename = f"{self.data_root}/val2014/{_filename}"
        for x in self.annotation:
            if 'COCO_val2014_' + str(x['image_id']).zfill(12) + '.jpg' == _filename:
                caption = x['caption']
                break
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        tokens = torch.cat((tokens, torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        gt = torch.clone(tokens)
        # if open-book
        if self.open_book:
            cap_stc = caption
            is_noun = lambda pos: pos[:2] == 'NN'
            is_verb = lambda pos: pos[:2] == 'VB'
            english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', 'A', 'The', 'To', 'An']
            stop_words = nltk.corpus.stopwords.words("english")
            cap_words = nltk.word_tokenize(cap_stc)
            cap_kws = [word for (word, pos) in nltk.pos_tag(cap_words) if ((word not in english_punctuations) and (word not in stop_words) and (is_noun or is_verb))]
            if not cap_kws:
                cap_kws.append(' ')
            kw_list = random.sample(cap_kws, random.choice(range(1, len(cap_kws))))
            kw = ''
            for key in kw_list:
                kw = kw + key + ','
        else:
            kw = ','
        kt = torch.tensor(self.tokenizer.encode(kw))

        pad_kt = self.max_key_len - kt.shape[0]
        if pad_kt > 0:
            kt = torch.cat((kt, torch.zeros(pad_kt, dtype=torch.int64) - 1))
        elif pad_kt < 0:
            kt = kt[:self.max_key_len]
        mask_kt = kt.ge(0)
        kt[~mask_kt] = 0
        mask_kt = mask_kt.float()

        padding = self.max_seq_len - tokens.shape[0]
        pad_gt = self.max_seq_len - gt.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        if pad_gt > 0:
            gt = torch.cat((gt, torch.zeros(pad_gt, dtype=torch.int64) - 1))  # we set target == -1 as ignore target
        elif pad_gt < 0:
            gt = gt[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()

        # mask = torch.cat((mask_kt, mask), dim=0)
        image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        return image, _filename, tokens, gt, mask, kt, kw

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.annotation = json.load(open("../dataset/MSCOCO_Caption/annotations/captions_val2014.json", "r"))["annotations"]
        with open('utils/captioneval/coco_test.txt') as f:
            self.files = f.read().splitlines()
        self.preprocess = _transform(224)
        self.max_seq_len = 20
        self.max_key_len = 20
        self.open_book = False


def alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.tag}-args.json")
    if args.local_rank == 0:
        with open(out_path, 'w') as outfile:
            json.dump(config, outfile)


def load_model(model, args, epoch_or_latest: Union[str, int] = '_latest'):
    # with open(config_path) as f:
    #     config = json.load(f)
    # parser = argparse.ArgumentParser()
    # parser.set_defaults(**config)
    # args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.tag}{epoch_or_latest}.pt")
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["model"])
    else:
        print(f"{model_path} is not exist")
    return model


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, clip_lr, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    print(f'No decay params: {no_decay_name}')
    print(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}
            ]


@torch.no_grad()
def val(model, epoch, val_dataloader, args):
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f">>> Evaling epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(val_dataloader), desc=args.tag)
    result_all = []
    val_loss_all = {}
    for i_vl in range(model.module.time_step):
        val_loss_all[f'vl_loss_{i_vl}'] = []
    for idx, (image, image_path, tokens, gt, mask, kt, kw) in enumerate(val_dataloader):
        image = image.cuda(non_blocking=True)
        tokens = tokens.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        kt = kt.cuda(non_blocking=True)
        # tokens_in = torch.cat((kt, tokens), dim=1)
        for i_t in range(0, model.module.time_step):
            b, device = mask.size()[0], mask.device
            t = torch.tensor(i_t, device=device).long().repeat_interleave(repeats=b)
            log_x_start = index_to_log_onehot(tokens, model.module.num_classes)
            log_xt = model.module.q_sample(log_x_start=log_x_start, t=t)
            xt = log_onehot_to_index(log_xt)
            # xt = torch.cat((kt, xt), dim=1)
            mask_tokens = xt
            ex_mask = torch.zeros_like(mask_tokens) - 10000
            ex_nomask = torch.zeros_like(mask_tokens)
            all_mask = torch.where(mask_tokens == 50257, ex_mask, ex_nomask)
            all_mask = all_mask.unsqueeze(dim=1).repeat_interleave(repeats=mask_tokens.size(1), dim=1)
            for each_b in range(all_mask.size(0)):
                for each_token in range(all_mask[each_b].size(0)):
                    all_mask[each_b, each_token, each_token] = 0
            padding_mask = mask.unsqueeze(dim=1)
            padding_mask = (1.0 - padding_mask.long()) * -10000
            all_mask = torch.clamp(all_mask + padding_mask, -10000, 0)
            outputs, len_out = model(tokens, mask_tokens, image, kt, all_mask, t)
            logits = outputs.logits
            loss_len = nnf.cross_entropy(len_out, mask.sum(dim=-1).to(torch.long) - 1)
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt.flatten(), ignore_index=-1)
            val_loss_all[f'vl_loss_{i_t}'].append(loss)
        prefix, len_cls = model.module.image_encode(image)
        prefix_embed = model.module.clip_project(prefix)
        kt = model.module.gpt.transformer.wte(kt)
        prefix_embed = model.module.kw_att(prefix_embed, kt)
        len_pre = model.module.len_head(len_cls)
        if args.use_beam_search:
            assert False, "Not check beam search for now"
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2_adpt_if(model, tokenizer, embed=prefix_embed, len_pre=len_pre.argmax(-1) + 1)

        torch.cuda.synchronize()
        progress.update()

        r = [{'image_id': _image_path, 'keyword': _kw, 'result': _text} for _image_path, _kw, _text in
             zip(image_path, kw, generated_text_prefix)]
        result_all.extend(r)
    progress.close()
    os.makedirs(f'.cache/{args.tag}', exist_ok=True)
    json.dump(result_all, open(f".cache/{args.tag}/tmp-results-{dist.get_rank()}.json", "w"))
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        result_all = []
        ra_id = []
        for i in range(dist.get_world_size()):
            part_result = json.load(open(f".cache/{args.tag}/tmp-results-{i}.json"))
            for ep in part_result:
                if ep['image_id'] not in ra_id:
                    ra_id.append(ep['image_id'])
                    result_all.append(ep)
        result = evaluate_on_coco_caption(result_all,
                                          os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}-results.json"),
                                          os.path.join(args.data_root, 'annotations/captions_val2014.json'))
    else:
        result = None
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        log_dict = {}
        for key in val_loss_all.keys():
            log_dict[key] = torch.tensor(val_loss_all[key]).mean().item()
        print(log_dict)
    return result


def sample_time(b, num_timesteps, device):
    t = torch.randint(0, num_timesteps, (b,), device=device).long()
    pt = torch.ones_like(t).float() / num_timesteps
    return t, pt


def multinomial_kl(log_prob1, log_prob2):  # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args,
          output_dir: str = ".", output_prefix: str = ""):
    model.train()
    num_steps = len(train_dataloader)

    train_dataloader.sampler.set_epoch(epoch)
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(train_dataloader), desc=output_prefix)
    for idx, (tokens, mask, prefix, gt, kt) in enumerate(train_dataloader):
        # prefix is raw images
        tokens = tokens.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        prefix = prefix.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        kt = kt.cuda(non_blocking=True)
        b, device = mask.size()[0], mask.device
        t, pt = sample_time(b, torch.tensor(args.time_step).to(torch.int), device)
        # add noise
        log_x_start = index_to_log_onehot(tokens, model.module.num_classes)
        log_xt = model.module.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)
        mask_tokens = xt
        # generate concentrate mask attention
        ex_mask = torch.zeros_like(mask_tokens) - 10000
        ex_nomask = torch.zeros_like(mask_tokens)
        all_mask = torch.where(mask_tokens == 50257, ex_mask, ex_nomask)
        all_mask = all_mask.unsqueeze(dim=1).repeat_interleave(repeats=mask_tokens.size(1), dim=1)
        for each_b in range(all_mask.size(0)):
            for each_token in range(all_mask[each_b].size(0)):
                all_mask[each_b, each_token, each_token] = 0
        padding_mask = mask.unsqueeze(dim=1)
        padding_mask = (1.0 - padding_mask.long()) * -10000
        all_mask = torch.clamp(all_mask + padding_mask, -10000, 0)

        # predict x0
        with amp.autocast(enabled=args.enable_amp):
            outputs, len_out = model(tokens, mask_tokens, prefix, kt, all_mask, t)
        logits = outputs.logits
        loss_len = nnf.cross_entropy(len_out, mask.sum(dim=-1).to(torch.long) - 1)
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt.flatten(), ignore_index=-1)
        loss = loss + loss_len
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()
        progress.set_postfix({"loss": loss.item() - loss_len.item(), 'lr': optimizer.param_groups[0]['lr'], 'loss_len':loss_len.item(), "loss_all": loss.item()})
        progress.update()
    progress.close()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../dataset/MSCOCO_Caption/oscar_split_ViT-B_32_train_512.pkl')
    parser.add_argument('--data_root', default='../dataset/MSCOCO_Caption/', help='raw coco training image path')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='debug',
                        help='tag of job, used for wandb and output')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--use_beam_search', action='store_true')
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--time_step',  type=int, default=20)
    parser.add_argument('--loss_weight', type=float, default=[0.5, 0.5])
    parser.add_argument('--if_drop_rate', type=float, default=0.1)
    parser.add_argument('--disable-amp', action='store_false', dest='enable_amp')
    parser.set_defaults(enable_amp=True)

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(args.out_dir, exist_ok=True)
    save_config(args)
    return args


def main(args):
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, Timestep=args.time_step,
                                  if_drop_rate=args.if_drop_rate)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type, Timestep=args.time_step,
                                 if_drop_rate=args.if_drop_rate)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params:,} total parameters')

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    # ckpt = torch.load('caption_diff_pretrain_CC-018.pt', map_location="cpu")
    # model.load_state_dict(ckpt["model"])
    model = model.cuda()

    parameters = get_pretrain_param_groups(model, args.lr * 0.1)
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    scaler = amp.GradScaler()
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    dataset = ClipCocoDataset(args.data_root, args.data, normalize_prefix=args.normalize_prefix)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=args.bs, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)

    val_dataset = ClipCocoValDataset(args.data_root)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=False)

    lr_args = {"LR_SCHEDULER_NAME": "cosine", "EPOCHS": args.epochs, "WARMUP_EPOCHS": 5, "MIN_LR": 1e-6,
               "WARMUP_LR": 1e-7}
    lr_scheduler = build_scheduler(lr_args, optimizer, len(train_dataloader))

    best_cider = 0
    for epoch in range(args.epochs):
        _ = train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args, output_dir=args.out_dir, output_prefix=args.tag)
        result = val(model, epoch, val_dataloader, args)
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if dist.get_rank() == 0:
                torch.save(
                    {'model':model.module.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()},
                    os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}.pt"),
                )
        if dist.get_rank() == 0 and result['CIDEr'] > best_cider:
            best_cider = result['CIDEr']
            torch.save(
                {'model':model.module.state_dict()},
                os.path.join(args.out_dir, f"{args.tag}-best.pt"),
            )

if __name__ == '__main__':
    # command:  python -m torch.distributed.launch --nproc_per_node 4 train.py --data ./oscar_split_ViT-B_32_train_512.pkl --out_dir ./output --bs 32
    args = parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    dist.init_process_group("nccl", init_method='env://', rank=args.local_rank, world_size=world_size)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)  ##### HERE

    seed = dist.get_rank() + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args)
