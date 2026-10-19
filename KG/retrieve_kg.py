"""Knowledge-graph retrieval and CLIP text embedding of the retrieved entities and relations.

This is the step that produces the "rel" features (data/<CORPUS>/features/<CORPUS>_rel_<phase>.hdf5).
The code that produced the released MSVD rel files (7 Jul 2025) is NOT in the repository: STGraph/clip_embedding.ipynb
only contains the text-embedding cell and an empty `res = {}` placeholder. This script re-implements the procedure
described in Section 3.5.2 of the paper (Eqs. 10 to 12) and fills that placeholder:

  1. every entity label e_i of V_K is encoded with the CLIP text encoder of the fine-tuned CLIP4Clip model (u_i),
  2. every video is sampled at 1 fps, uniformly reduced to T = 20 frames, each frame is encoded with the CLIP image
     encoder (v_t), and s_i = cos(v_t, u_i) is aggregated over the frames (max by default, --agg mean for the mean),
  3. the top-k entities form R_o (k = 10 by default),
  4. R_r = {(h, r, t) in R_K : h in R_o and t in R_o} (Eq. 12); with --rel_scope any, one endpoint in R_o is enough,
  5. the k entity strings and the |R_r| relation strings are encoded with the CLIP text encoder, L2-normalised and
     concatenated into a (k + |R_r|) x 512 matrix that is stored under the video id, exactly like the notebook cell.
     --relation_text triple encodes "h r t" (the example in the notebook), --relation_text relation encodes only r.

Validate against the released files before producing new features: --check data/MSVD/features/MSVD_rel_test.hdf5
prints, for every processed video that exists in that file, the number of rows there and here and the fraction of
released rows that have a recomputed row with cosine similarity > 0.99. If the agreement is low, the original
retrieval used a different aggregation, k, or relation text: try --agg mean, --relation_text relation, --rel_scope any,
or ask the author of the original run.

Usage (from the repository root, after STGraph/modules and STGraph/pretrained are in place):
    python KG/retrieve_kg.py --dataset MSVD --videos data/MSVD/raw --triples KG/msvd_filtered_entities.txt \
        --mapping data/MSVD/captions/youtube_mapping.txt --clip4clip STGraph/pretrained/pytorch_model.bin.0 \
        --k 10 --out KG/msvd_rel_all.hdf5 --res_json KG/msvd_res.json [--check data/MSVD/features/MSVD_rel_test.hdf5]
    python KG/retrieve_kg.py --dataset MSR-VTT --videos data/MSR-VTT/videos/all --triples KG/msrvtt_filtered_entities.txt \
        --clip4clip CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.4 --k 10 --out KG/msrvtt_rel_all.hdf5
Then split the output with split/split_features.py --rel KG/<...>_rel_all.hdf5.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
STGRAPH = os.path.join(os.path.dirname(HERE), 'STGraph')
sys.path.insert(0, STGRAPH)


# ----------------------------------------------------------------------------------------------------------------
# Knowledge graph
# ----------------------------------------------------------------------------------------------------------------
def load_kg(path):
    """<dataset>_filtered_entities.txt: one 'head &tail &relation' per line."""
    triples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.split('&')]
            if len(parts) == 3 and all(parts):
                triples.append(tuple(parts))
    entities = sorted({h for h, _, _ in triples} | {t for _, t, _ in triples})
    by_pair = defaultdict(list)     # (h, t) -> [r, ...]
    by_node = defaultdict(list)     # h or t -> [(h, t, r), ...]
    for h, t, r in triples:
        by_pair[(h, t)].append(r)
        by_node[h].append((h, t, r))
        if t != h:
            by_node[t].append((h, t, r))
    return triples, entities, by_pair, by_node


def retrieve_relations(R_o, by_pair, by_node, scope='both', text='triple'):
    chosen, seen = [], set()
    if scope == 'both':
        for h in R_o:
            for t in R_o:
                for r in by_pair.get((h, t), []):
                    key = (h, t, r)
                    if key not in seen:
                        seen.add(key)
                        chosen.append(key)
    else:
        for e in R_o:
            for key in by_node.get(e, []):
                if key not in seen:
                    seen.add(key)
                    chosen.append(key)
    if text == 'triple':
        strings = [f'{h} {r} {t}' for h, t, r in chosen]
    else:
        strings = []
        for _, _, r in chosen:
            if r not in strings:
                strings.append(r)
    return chosen, strings


# ----------------------------------------------------------------------------------------------------------------
# CLIP4Clip model, tokenizer and frames (same code path as STGraph/grid_node_feature_extractor.py)
# ----------------------------------------------------------------------------------------------------------------
class TaskArgs:
    slice_framepos = 2
    max_frames = 20
    eval_frame_order = 0
    cache_dir = ''
    max_words = 32
    feature_framerate = 1
    cross_model = 'cross-base'
    local_rank = 0
    pretrained_clip_name = 'ViT-B/16'   # ViT-B/32 if the CLIP4Clip checkpoint was trained with ViT-B/32


def load_clip(ckpt, device, clip_name):
    from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
    from modules.modeling import CLIP4Clip
    TaskArgs.pretrained_clip_name = clip_name
    state = torch.load(ckpt, map_location='cpu')
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(TaskArgs.cross_model, cache_dir=cache_dir, state_dict=state, task_config=TaskArgs)
    clip = model.clip.to(device).eval()
    return clip


class ClipTokenizer:
    """CLIP BPE tokenizer of CLIP4Clip (identical ids to open_clip's ViT-B-32 tokenizer used in the notebook)."""

    def __init__(self, context_length=77):
        from modules.tokenization_clip import SimpleTokenizer
        self.tok = SimpleTokenizer()
        self.sot = self.tok.encoder['<|startoftext|>']
        self.eot = self.tok.encoder['<|endoftext|>']
        self.n = context_length

    def __call__(self, texts):
        out = torch.zeros(len(texts), self.n, dtype=torch.long)
        for i, s in enumerate(texts):
            ids = [self.sot] + self.tok.encode(s)[: self.n - 2] + [self.eot]
            out[i, :len(ids)] = torch.tensor(ids)
        return out


@torch.no_grad()
def encode_texts(clip, tokenizer, texts, device, batch=256):
    outs = []
    for i in range(0, len(texts), batch):
        tok = tokenizer(texts[i:i + batch]).to(device)
        x = clip.encode_text(tok).float()
        outs.append(torch.nn.functional.normalize(x, dim=-1).cpu())
    return torch.cat(outs) if outs else torch.zeros(0, 512)


def make_frame_reader(max_frames):
    from dataloaders.rawvideo_util import RawVideoExtractorCV2
    ext = RawVideoExtractorCV2(size=224, framerate=1, type=0)

    def read(path):
        data, _ = ext.get_video_data(path, patch=0)
        frames = data['video']
        if frames.dim() < 4:
            return None
        frames = frames.reshape(-1, *frames.shape[-3:])          # (L, 3, 224, 224)
        if frames.shape[0] > max_frames:                          # slice_framepos = 2: uniform sampling
            idx = np.linspace(0, frames.shape[0] - 1, num=max_frames, dtype=int)
            frames = frames[idx]
        return frames.float()
    return read


@torch.no_grad()
def encode_frames(clip, frames, device):
    x = clip.encode_image(frames.to(device), video_frame=frames.shape[0]).float()
    return torch.nn.functional.normalize(x, dim=-1).cpu()


# ----------------------------------------------------------------------------------------------------------------
def list_videos(dataset, videos_dir, mapping_path):
    files = sorted(glob.glob(os.path.join(videos_dir, '*')))
    files = [f for f in files if os.path.splitext(f)[1].lower() in ('.avi', '.mp4', '.mkv', '.webm', '.mov')]
    if dataset == 'MSVD':
        name2vid = {}
        with open(mapping_path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    a, b = p[0], p[-1]
                    if a.startswith('vid') and a[3:].isdigit():
                        name2vid[b] = a
                    else:
                        name2vid[a] = b
        items = []
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            if name in name2vid:
                items.append((name2vid[name], f))
        return items
    return [(os.path.splitext(os.path.basename(f))[0], f) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['MSVD', 'MSR-VTT'])
    ap.add_argument('--videos', required=True, help='directory with the raw clips (MSVD .avi, MSR-VTT videoN.mp4)')
    ap.add_argument('--triples', required=True, help='KG/<dataset>_filtered_entities.txt')
    ap.add_argument('--mapping', default=None, help='MSVD youtube_mapping.txt (vidN <clip name>)')
    ap.add_argument('--clip4clip', required=True, help='fine-tuned CLIP4Clip state dict (pytorch_model.bin.N)')
    ap.add_argument('--clip_name', default='ViT-B/16')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--agg', default='max', choices=['max', 'mean'], help='aggregation of cos(v_t, u_i) over frames')
    ap.add_argument('--rel_scope', default='both', choices=['both', 'any'])
    ap.add_argument('--relation_text', default='triple', choices=['triple', 'relation'])
    ap.add_argument('--max_frames', type=int, default=20)
    ap.add_argument('--out', required=True, help='hdf5 with one (k + |R_r|) x 512 dataset per video id')
    ap.add_argument('--res_json', default=None, help='json with the retrieved entities/relations per video')
    ap.add_argument('--check', default=None, help='released rel hdf5 to compare with (e.g. MSVD_rel_test.hdf5)')
    ap.add_argument('--limit', type=int, default=0, help='process only the first N videos (debugging)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    triples, entities, by_pair, by_node = load_kg(args.triples)
    print(f'KG: {len(triples)} triples, {len(entities)} entities, {len({r for _, _, r in triples})} relation types')

    clip = load_clip(args.clip4clip, device, args.clip_name)
    tokenizer = ClipTokenizer()
    print('encoding entity labels ...')
    U = encode_texts(clip, tokenizer, entities, device)           # |V_K| x 512, L2-normalised
    read_frames = make_frame_reader(args.max_frames)

    items = list_videos(args.dataset, args.videos, args.mapping)
    if args.limit:
        items = items[:args.limit]
    print(f'{len(items)} videos')

    ref = h5py.File(args.check, 'r') if args.check else None
    agree, n_checked = [], 0
    res = {}
    with h5py.File(args.out, 'w') as fo:
        for vid, path in tqdm(items):
            frames = read_frames(path)
            if frames is None:
                print('cannot read', path)
                continue
            V = encode_frames(clip, frames, device)                 # T x 512
            S = V @ U.T                                            # T x |V_K|
            score = S.max(0).values if args.agg == 'max' else S.mean(0)
            top = torch.topk(score, min(args.k, len(entities))).indices.tolist()
            R_o = [entities[i] for i in top]
            chosen, rel_strings = retrieve_relations(R_o, by_pair, by_node, args.rel_scope, args.relation_text)
            E = encode_texts(clip, tokenizer, R_o, device)
            R = encode_texts(clip, tokenizer, rel_strings, device) if rel_strings else torch.zeros(0, E.shape[1])
            emb = torch.cat([E, R]).numpy().astype(np.float32)
            fo.create_dataset(vid, data=emb)
            res[vid] = {'entities': R_o, 'scores': [float(score[i]) for i in top],
                        'relations': rel_strings, 'triples': [list(t) for t in chosen]}
            if ref is not None and vid in ref:
                old = torch.tensor(ref[vid][()]).float()
                old = torch.nn.functional.normalize(old, dim=-1)
                sim = (old @ torch.tensor(emb).T).max(1).values
                frac = float((sim > 0.99).float().mean())
                agree.append(frac)
                n_checked += 1
                if n_checked <= 10:
                    print(f'{vid}: released rows {old.shape[0]}, recomputed rows {emb.shape[0]}, matched {frac:.2f}')
    if ref is not None:
        ref.close()
        if agree:
            print(f'checked {n_checked} videos: mean fraction of released rows reproduced = {np.mean(agree):.3f}')
    if args.res_json:
        with open(args.res_json, 'w') as f:
            json.dump(res, f)
    print('saved', args.out)


if __name__ == '__main__':
    main()
