# coding=utf-8
import inspect
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from model.model import pad_mask
from model.label_smoothing import LabelSmoothing
import numpy as np
from torch_geometric.data import Data

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import gc


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [[] for _ in range(self.num_losses)]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [0. for _ in range(self.num_losses)]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def graph_batch_offset(geo_x, geo_edge_index, i, C):
    """Offset added to the edge indices of the i-th graph of a mini-batch.

    The node features of the batch are stacked as (batch * n_node, dim), so the nodes of graph i occupy the rows
    [i * n_node, (i + 1) * n_node) and the offset is n_node * i (n_node = geo_x.shape[1] = T * G^2 = 180).
    The experiments reported in the paper were run with the offset inherited from the AKVA code base,
    int(sqrt(n_edges)) * i, which is kept behind the --legacy_batching flag so that those runs can be reproduced.
    """
    if getattr(C, 'legacy_batching', False):
        return int(np.sqrt(geo_edge_index[0].shape[1])) * i
    return geo_x.shape[1] * i


def parse_batch(batch, feature_mode, device):
    vids = batch[0]
    if feature_mode == 'grid-obj-rel':
        video_masks, geo_x_list, edge_index_list, edge_attr_list, object_feats, rel_feats, r2l_captions, l2r_captions = batch[1:]
    elif feature_mode == 'grid-rel':
        video_masks, geo_x_list, edge_index_list, edge_attr_list, rel_feats, r2l_captions, l2r_captions = batch[1:]
        object_feats = None
    elif feature_mode == 'grid':
        video_masks, geo_x_list, edge_index_list, edge_attr_list, r2l_captions, l2r_captions = batch[1:]
        object_feats = rel_feats = None
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    geo_x_feats = geo_x_list.to(device, non_blocking=True)
    geo_edge_index_feats = edge_index_list.to(device, non_blocking=True)
    geo_edge_attr_feats = edge_attr_list.to(device, non_blocking=True)
    video_mask_feats = torch.cat(video_masks, dim=1).to(device, non_blocking=True)
    del geo_x_list, edge_index_list, edge_attr_list, video_masks

    if object_feats is not None:
        object_feats = object_feats.to(device, non_blocking=True)
    if rel_feats is not None:
        rel_feats = rel_feats.to(device, non_blocking=True)

    r2l_captions = r2l_captions.to(device, non_blocking=True)
    l2r_captions = l2r_captions.to(device, non_blocking=True)

    if feature_mode == 'grid-obj-rel':
        feats = (geo_x_feats, geo_edge_index_feats, geo_edge_attr_feats, object_feats, rel_feats, video_mask_feats)
    elif feature_mode == 'grid-rel':
        feats = (geo_x_feats, geo_edge_index_feats, geo_edge_attr_feats, rel_feats, video_mask_feats)
    elif feature_mode == 'grid':
        feats = (geo_x_feats, geo_edge_index_feats, geo_edge_attr_feats, video_mask_feats)

    gc.collect()
    return vids, feats, r2l_captions, l2r_captions


def train(e, model, optimizer, train_iter, vocab, reg_lambda, gradient_clip, feature_mode, lr_scheduler, C, device, local_rank):
    torch.cuda.empty_cache()
    model.train()
    loss_checker = LossChecker(3)
    pad_idx = vocab.word2idx['<PAD>']
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    if local_rank == 0:
        t = tqdm(train_iter)
    else:
        t = train_iter
    for step, batch in enumerate(t):
        _, feats, r2l_captions, l2r_captions = parse_batch(batch, feature_mode, device)

        r2l_trg = r2l_captions[:, :-1]
        r2l_trg_y = r2l_captions[:, 1:]
        l2r_trg = l2r_captions[:, :-1]
        l2r_trg_y = l2r_captions[:, 1:]

        if feature_mode == 'grid-obj-rel':
            geo_x, geo_edge_index, geo_edge_attr, object_feats, rel_feats, video_mask = feats
            batch_sz = geo_x.shape[0]
            # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
            x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

            # Tính toán offset cho geo_edge_index
            offset = []
            # Duyệt qua từng mẫu trong batch
            for i in range(batch_sz):
                offset_val = graph_batch_offset(geo_x, geo_edge_index, i, C)
                offset.append(torch.full(geo_edge_index[0].shape, offset_val))
            offset = torch.stack(offset).cuda()
            # Cộng offset vào geo_edge_index để tạo batch offset
            geo_graph_batch_offset = geo_edge_index + offset

            # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
            new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
            edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

            # Reshape geo_edge_attr theo định dạng của code bên dưới
            edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                    geo_edge_attr.shape[2]).float()

            # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
            data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
            feats = (data_geo_graph_batch, object_feats, rel_feats)
            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
        elif feature_mode == 'grid-rel':
            geo_x, geo_edge_index, geo_edge_attr, rel_feats, video_mask = feats
            batch_sz = geo_x.shape[0]
            # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
            x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

            # Tính toán offset cho geo_edge_index
            offset = []
            # Duyệt qua từng mẫu trong batch
            for i in range(batch_sz):
                offset_val = graph_batch_offset(geo_x, geo_edge_index, i, C)
                offset.append(torch.full(geo_edge_index[0].shape, offset_val))
            offset = torch.stack(offset).cuda()
            # Cộng offset vào geo_edge_index để tạo batch offset
            geo_graph_batch_offset = geo_edge_index + offset

            # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
            new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
            edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

            # Reshape geo_edge_attr theo định dạng của code bên dưới
            edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                    geo_edge_attr.shape[2]).float()

            # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
            data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
            feats = (data_geo_graph_batch, rel_feats)
            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
        elif feature_mode == 'grid': 
            geo_x, geo_edge_index, geo_edge_attr, video_mask = feats
            batch_sz = geo_x.shape[0]
            # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
            x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

            # Tính toán offset cho geo_edge_index
            offset = []
            # Duyệt qua từng mẫu trong batch
            for i in range(batch_sz):
                offset_val = graph_batch_offset(geo_x, geo_edge_index, i, C)
                offset.append(torch.full(geo_edge_index[0].shape, offset_val))
            offset = torch.stack(offset).cuda()
            # Cộng offset vào geo_edge_index để tạo batch offset
            geo_graph_batch_offset = geo_edge_index + offset

            # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
            new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
            edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

            # Reshape geo_edge_attr theo định dạng của code bên dưới
            edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                    geo_edge_attr.shape[2]).float()

            # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
            data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
            feats = data_geo_graph_batch
            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)

        r2l_pred, l2r_pred = model(feats, r2l_trg, l2r_trg, mask)

        r2l_loss = criterion(r2l_pred.view(-1, vocab.n_vocabs),
                             r2l_trg_y.contiguous().view(-1)) 
        l2r_loss = criterion(l2r_pred.view(-1, vocab.n_vocabs),
                             l2r_trg_y.contiguous().view(-1))

        r2l_loss = r2l_loss /C.gradient_accumulation_steps
        l2r_loss = l2r_loss/C.gradient_accumulation_steps
        loss = reg_lambda * l2r_loss + (1 - reg_lambda) * r2l_loss
        loss.backward()
        if (step + 1) % C.gradient_accumulation_steps == 0:
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step() 
            optimizer.zero_grad()

        loss_checker.update(loss.item(), r2l_loss.item(), l2r_loss.item())

        # del r2l_trg, r2l_trg_y, l2r_trg, l2r_trg_y, r2l_pred, l2r_pred, mask, feats
        # if feature_mode.startswith('grid'):
        #     del data_geo_graph_batch, geo_graph_batch_offset, geo_graph_batch_offset, edge_attr_batch, edge_index_batch, x_batch
        torch.cuda.empty_cache()
        gc.collect()

        if local_rank == 0: 
            t.set_description("[Epoch #{0}] loss: {3:.3f} = (reg: {1:.3f} * r2l_loss: {4:.3f} + "
                            "(1 - reg): {2:.3f} * l2r_loss: {5:.3f})"
                            .format(e, 1 - reg_lambda, reg_lambda, *loss_checker.mean(last=10)))
    
    total_loss, r2l_loss, l2r_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'r2l_loss': r2l_loss,
        'l2r_loss': l2r_loss
    }
    return loss

def test(model, val_iter, vocab, reg_lambda, feature_mode, C, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()

    loss_checker = LossChecker(3)
    pad_idx = vocab.word2idx['<PAD>']
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    with torch.no_grad():
        for batch in tqdm(val_iter, desc='Test'):
            _, feats, r2l_captions, l2r_captions = parse_batch(batch, feature_mode, device)
            
            r2l_trg = r2l_captions[:, :-1]
            r2l_trg_y = r2l_captions[:, 1:]
            l2r_trg = l2r_captions[:, :-1]
            l2r_trg_y = l2r_captions[:, 1:]

            if feature_mode == 'grid-obj-rel':
                geo_x, geo_edge_index, geo_edge_attr, object_feats, rel_feats, video_mask = feats
                batch_sz = geo_x.shape[0]
                # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
                x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

                # Tính toán offset cho geo_edge_index
                offset = []
                # Duyệt qua từng mẫu trong batch
                for i in range(batch_sz):
                    offset_val = graph_batch_offset(geo_x, geo_edge_index, i, C)
                    offset.append(torch.full(geo_edge_index[0].shape, offset_val))
                offset = torch.stack(offset).cuda()
                # Cộng offset vào geo_edge_index để tạo batch offset
                geo_graph_batch_offset = geo_edge_index + offset

                # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
                new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
                edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

                # Reshape geo_edge_attr theo định dạng của code bên dưới
                edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                        geo_edge_attr.shape[2]).float()

                # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
                data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
                feats = (data_geo_graph_batch, object_feats, rel_feats)
                mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
            elif feature_mode == 'grid-rel':
                geo_x, geo_edge_index, geo_edge_attr, rel_feats, video_mask = feats
                batch_sz = geo_x.shape[0]
                # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
                x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

                # Tính toán offset cho geo_edge_index
                offset = []
                # Duyệt qua từng mẫu trong batch
                for i in range(batch_sz):
                    offset_val = graph_batch_offset(geo_x, geo_edge_index, i, C)
                    offset.append(torch.full(geo_edge_index[0].shape, offset_val))
                offset = torch.stack(offset).cuda()
                # Cộng offset vào geo_edge_index để tạo batch offset
                geo_graph_batch_offset = geo_edge_index + offset

                # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
                new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
                edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

                # Reshape geo_edge_attr theo định dạng của code bên dưới
                edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                        geo_edge_attr.shape[2]).float()

                # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
                data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
                feats = (data_geo_graph_batch, rel_feats)
                mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
            elif feature_mode == 'grid': 
                geo_x, geo_edge_index, geo_edge_attr, video_mask = feats
                batch_sz = geo_x.shape[0]
                # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
                x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

                # Tính toán offset cho geo_edge_index
                offset = []
                # Duyệt qua từng mẫu trong batch
                for i in range(batch_sz):
                    offset_val = graph_batch_offset(geo_x, geo_edge_index, i, C)
                    offset.append(torch.full(geo_edge_index[0].shape, offset_val))
                offset = torch.stack(offset).cuda()
                # Cộng offset vào geo_edge_index để tạo batch offset
                geo_graph_batch_offset = geo_edge_index + offset

                # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
                new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
                edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

                # Reshape geo_edge_attr theo định dạng của code bên dưới
                edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                        geo_edge_attr.shape[2]).float()

                # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
                data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
                feats = data_geo_graph_batch
                mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)


            r2l_pred, l2r_pred = model(feats, r2l_trg, l2r_trg, mask)

            r2l_loss = criterion(r2l_pred.view(-1, vocab.n_vocabs),
                                r2l_trg_y.contiguous().view(-1))
            l2r_loss = criterion(l2r_pred.view(-1, vocab.n_vocabs),
                                l2r_trg_y.contiguous().view(-1))
            loss = reg_lambda * l2r_loss + (1 - reg_lambda) * r2l_loss
            loss_checker.update(loss.item(), r2l_loss.item(), l2r_loss.item())

            # del r2l_trg, r2l_trg_y, l2r_trg, l2r_trg_y, r2l_pred, l2r_pred, mask, feats
            # if feature_mode.startswith('grid'):
            #     del data_geo_graph_batch, geo_graph_batch_offset, geo_graph_batch_offset, edge_attr_batch, edge_index_batch, x_batch
            torch.cuda.empty_cache()
            gc.collect()

        total_loss, r2l_loss, l2r_loss = loss_checker.mean()
        loss = {
            'total': total_loss,
            'r2l_loss': r2l_loss,
            'l2r_loss': l2r_loss
        }
    return loss

def get_predicted_captions(data_iter, model, beam_size, max_len, feature_mode, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    r2l_vid2pred = {}
    l2r_vid2pred = {}
    vids_set = set()
    with torch.no_grad():
        for batch in tqdm(iter(data_iter), desc='get_predicted_captions'):
            vids, video_masks = batch[:2]
            video_masks = video_masks[0]

            if feature_mode == 'grid-obj-rel':
                geo_x_list, edge_index_list, edge_attr_list, object_feats, rel_feats, r2l_captions, l2r_captions = batch[2:]
            elif feature_mode == 'grid-rel':
                geo_x_list, edge_index_list, edge_attr_list, rel_feats, r2l_captions, l2r_captions = batch[2:]
                object_feats = None
            elif feature_mode == 'grid':
                geo_x_list, edge_index_list, edge_attr_list, r2l_captions, l2r_captions = batch[2:]
                object_feats = None
                rel_feats = None
            else:
                raise NotImplementedError
            
            for i, vid in enumerate(vids):
                if vid not in vids_set:
                    vids_set.add(vid)
                    data_geo_graph = Data(x=geo_x_list[i].to(device), edge_index=edge_index_list[i].to(device), edge_attr=edge_attr_list[i].to(device))
                    if object_feats is not None and rel_feats is not None:
                        object_feat = object_feats[i].unqueeze(0)
                        rel_feat = rel_feats[i].unsqueeze(0)
                        feats = (data_geo_graph, object_feat.to(device), rel_feat.to(device))
                    elif rel_feats is not None:
                        rel_feat = rel_feats[i].unsqueeze(0)
                        feats = (data_geo_graph, rel_feat.to(device))
                    else:
                        feats = (data_geo_graph)

                    r2l_captions, l2r_captions = model.beam_search_decode(feats, beam_size, max_len, video_masks[i].unsqueeze(0).to(device))
                    l2r_captions = [" ".join(caption[0].value) for caption in l2r_captions]
                    r2l_captions = [" ".join(caption[0].value) for caption in r2l_captions]
                    r2l_vid2pred[vid] = r2l_captions[0]
                    l2r_vid2pred[vid] = l2r_captions[0]
                else: 
                    continue
                torch.cuda.empty_cache()
                gc.collect()

    return r2l_vid2pred, l2r_vid2pred


def get_groundtruth_captions(data_iter, vocab, feature_mode):
    r2l_vid2GTs = {}
    l2r_vid2GTs = {}
    S_idx = vocab.word2idx['<S>']
    for batch in tqdm(iter(data_iter), desc='get_groundtruth_captions'):
        vids = batch[0]
        r2l_captions = batch[-2]
        l2r_captions = batch[-1]
        for vid, r2l_caption, l2r_caption in zip(vids, r2l_captions, l2r_captions):
            if vid not in r2l_vid2GTs:
                r2l_vid2GTs[vid] = []
            if vid not in l2r_vid2GTs:
                l2r_vid2GTs[vid] = []
            r2l_caption = idxs_to_sentence(r2l_caption, vocab.idx2word, S_idx)
            l2r_caption = idxs_to_sentence(l2r_caption, vocab.idx2word, S_idx)
            r2l_vid2GTs[vid].append(r2l_caption)
            l2r_vid2GTs[vid].append(l2r_caption)
        del r2l_captions, l2r_captions, vids
        gc.collect()
    return r2l_vid2GTs, l2r_vid2GTs


def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = {v: i for i, v in enumerate(vid2pred.keys())}
    refs = {vid2idx[vid]: GTs for vid, GTs in vid2GTs.items()}
    hypos = {vid2idx[vid]: [pred] for vid, pred in vid2pred.items()}

    scores = calc_scores(refs, hypos)
    return scores

def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = float(score)
    return final_scores

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls

def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'vc_model': model.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([str(vid), pred, GTs])
            fout.write("{}\n".format(line))