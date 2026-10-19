"""Split the extracted features into the train / val / test files expected by config.py.

One script for both corpora and the three feature types.

Output names follow config.py:
    <root>/data/<CORPUS>/features/<CORPUS>_GBased_<phase>.pickle
    <root>/data/<CORPUS>/features/<CORPUS>_rel_<phase>.hdf5
    <root>/data/<CORPUS>/features/<CORPUS>_videomask_<phase>.hdf5
where CORPUS is exactly "MSVD" or "MSR-VTT" (note the hyphen: the Drive folder is called MSRVTT and must be renamed
or symlinked to MSR-VTT, because TrainConfig builds the paths from the model name prefix).

Split definition:
    MSVD     : data/MSVD/metadata/{train,val,test}.list (JSON lists of "vidN") + data/MSVD/captions/youtube_mapping.txt
               ("vidN  <youtube_id>_<start>_<end>" per line) because the STGraph files are keyed by the YouTube clip name.
    MSR-VTT  : standard split by video number: video0..video6512 train, video6513..video7009 val, video7010..video9999 test
               (the same 6513 / 497 / 2990 counts as MSRVTTLoaderConfig).

Usage (run from the repository root):
    python split/split_features.py --dataset MSVD --root . \
        --stg STGraph/extracted/msvd/stg.pickle --vidmask STGraph/extracted/msvd/vid_mask.hdf5 [--rel KG/msvd_rel_all.hdf5]
    python split/split_features.py --dataset MSR-VTT --root . \
        --stg STGraph/extracted/msrvtt_grid/stg.pickle --vidmask STGraph/extracted/msrvtt_grid/video_mask.hdf5 [--rel KG/msrvtt_rel_all.hdf5]

Any of --stg / --vidmask / --rel may be omitted. Existing output files are not overwritten unless --overwrite is given.
Memory: the GBased pickle is loaded completely (the MSVD one is about 37 GB on disk), so run this on a machine with
enough RAM, or pass --stg_lowmem to stream the pickle one phase at a time (three passes, three times slower).
"""
import argparse
import json
import os
import pickle
import re
import sys

import h5py
from tqdm import tqdm


def load_msvd_split(root):
    meta = os.path.join(root, 'data', 'MSVD', 'metadata')
    lists = {}
    for phase in ['train', 'val', 'test']:
        with open(os.path.join(meta, f'{phase}.list')) as f:
            lists[phase] = set(json.load(f))
    mapping = {}
    with open(os.path.join(root, 'data', 'MSVD', 'captions', 'youtube_mapping.txt')) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[-1]
            # accept both "vidN name" and "name vidN"
            if a.startswith('vid') and a[3:].isdigit():
                mapping[b] = a
            else:
                mapping[a] = b
    return lists, mapping


def msrvtt_phase(vid):
    m = re.match(r'video(\d+)$', vid)
    if not m:
        return None
    n = int(m.group(1))
    if n <= 6512:
        return 'train'
    if n <= 7009:
        return 'val'
    if n <= 9999:
        return 'test'
    return None


def resolve(key, dataset, lists, mapping):
    """Return (video_id_used_by_the_loader, phase) or (None, None)."""
    if dataset == 'MSVD':
        vid = key if (key.startswith('vid') and key[3:].isdigit()) else mapping.get(key)
        if vid is None:
            return None, None
        for phase, s in lists.items():
            if vid in s:
                return vid, phase
        return vid, None
    base = key[:-4] if key.endswith('.mp4') else key
    return base, msrvtt_phase(base)


def out_path(root, corpus, feat, phase, ext):
    d = os.path.join(root, 'data', corpus, 'features')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{corpus}_{feat}_{phase}.{ext}')


def split_hdf5(src, root, corpus, feat, dataset, lists, mapping, overwrite):
    outs = {p: out_path(root, corpus, feat, p, 'hdf5') for p in ['train', 'val', 'test']}
    if not overwrite and all(os.path.exists(p) for p in outs.values()):
        print(f'[{feat}] outputs exist, skipping (use --overwrite)')
        return
    counts = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
    with h5py.File(src, 'r') as fin, \
            h5py.File(outs['train'], 'w') as ftr, h5py.File(outs['val'], 'w') as fva, h5py.File(outs['test'], 'w') as fte:
        fo = {'train': ftr, 'val': fva, 'test': fte}
        for key in tqdm(list(fin.keys()), desc=f'{feat}'):
            vid, phase = resolve(key, dataset, lists, mapping)
            if phase is None:
                counts['skipped'] += 1
                continue
            fo[phase].create_dataset(vid, data=fin[key][()])
            counts[phase] += 1
    print(f'[{feat}]', counts)


def split_pickle(src, root, corpus, dataset, lists, mapping, overwrite, lowmem):
    outs = {p: out_path(root, corpus, 'GBased', p, 'pickle') for p in ['train', 'val', 'test']}
    if not overwrite and all(os.path.exists(p) for p in outs.values()):
        print('[GBased] outputs exist, skipping (use --overwrite)')
        return
    counts = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
    if lowmem:
        for phase in ['train', 'val', 'test']:
            with open(src, 'rb') as f:
                data = pickle.load(f)
            part = {}
            for key, value in data.items():
                vid, ph = resolve(key, dataset, lists, mapping)
                if ph == phase:
                    part[vid] = value
            del data
            with open(outs[phase], 'wb') as f:
                pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)
            counts[phase] = len(part)
            del part
    else:
        with open(src, 'rb') as f:
            data = pickle.load(f)
        parts = {'train': {}, 'val': {}, 'test': {}}
        for key, value in tqdm(data.items(), desc='GBased'):
            vid, phase = resolve(key, dataset, lists, mapping)
            if phase is None:
                counts['skipped'] += 1
                continue
            parts[phase][vid] = value
        del data
        for phase, part in parts.items():
            with open(outs[phase], 'wb') as f:
                pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)
            counts[phase] = len(part)
    print('[GBased]', counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['MSVD', 'MSR-VTT'])
    ap.add_argument('--root', default='.', help='repository root that contains data/<CORPUS>/')
    ap.add_argument('--stg', default=None, help='stg.pickle produced by STGraph/transform-graph-to-geometric.py')
    ap.add_argument('--vidmask', default=None, help='vid_mask.hdf5 (MSVD) or video_mask.hdf5 (MSR-VTT) from grid_node_feature_extractor.py')
    ap.add_argument('--rel', default=None, help='hdf5 with the KG embeddings of every video (KG/retrieve_kg.py output)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--stg_lowmem', action='store_true')
    args = ap.parse_args()

    corpus = args.dataset
    if corpus == 'MSVD':
        lists, mapping = load_msvd_split(args.root)
        print({k: len(v) for k, v in lists.items()}, 'mapping entries:', len(mapping))
    else:
        lists, mapping = None, None
    if not os.path.isdir(os.path.join(args.root, 'data', corpus)):
        sys.exit(f'data/{corpus} not found under {args.root} (for MSR-VTT rename or symlink data/MSRVTT -> data/MSR-VTT)')

    if args.vidmask:
        split_hdf5(args.vidmask, args.root, corpus, 'videomask', corpus, lists, mapping, args.overwrite)
    if args.rel:
        split_hdf5(args.rel, args.root, corpus, 'rel', corpus, lists, mapping, args.overwrite)
    if args.stg:
        split_pickle(args.stg, args.root, corpus, corpus, lists, mapping, args.overwrite, args.stg_lowmem)


if __name__ == '__main__':
    main()
