from __future__ import print_function, division

from collections import defaultdict

from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms
import pickle
import gc

from loader.transform import UniformSample, RandomSample, ToTensor, TrimExceptAscii, Lowercase, \
    RemovePunctuation, SplitWithWhiteSpace, Truncate, PadFirst, PadLast, PadToLength, \
    ToIndex

class CustomVocab(object):
    def __init__(self, caption_fpath, init_word2idx, min_count=1, transform=str.split):
        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform

        self.word2idx = init_word2idx
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq_dict = defaultdict(lambda: 0)
        self.n_vocabs = len(self.word2idx)
        self.n_words = self.n_vocabs
        self.max_sentence_len = -1

        self.build()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build(self):
        captions = self.load_captions()
        for caption in captions:
            words = self.transform(caption)
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                self.word_freq_dict[word] += 1
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        keep_words = [word for word, freq in self.word_freq_dict.items() if freq >= self.min_count]

        for idx, word in enumerate(keep_words, len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([self.word_freq_dict[word] for word in keep_words])


class CustomDataset(Dataset):
    """ Dataset """

    def __init__(self, C, phase, caption_fpath, transform_caption=None, transform_frame=None):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption
        self.feature_mode = C.feat.feature_mode

        if self.feature_mode == 'grid-obj-rel':
            self.object_video_feats = defaultdict(lambda: [])
            self.rel_feats = defaultdict(lambda: [])
        elif self.feature_mode == 'grid-rel':
            self.rel_feats = defaultdict(lambda: [])

        self.video_mask = defaultdict(lambda: [])
        self.r2l_captions = defaultdict(lambda: [])
        self.l2r_captions = defaultdict(lambda: [])

        models = self.C.feat.model.split('_')[1].split('+')
        path = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus, f"{self.C.corpus}_{models[0]}",
                                                                    self.phase, 'pickle')
        with open(path, "rb") as fb:
            self.graph = pickle.load(fb)

        self.data = []
        self.build_video_caption_pairs()

        self.r2l_captions.clear()
        self.l2r_captions.clear()
        gc.collect()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid, r2l_caption, l2r_caption = self.data[idx]
        video_mask = self.video_mask[vid]

        stgraph = self.graph[vid]
        x = stgraph['x']
        pad_len = self.C.transformer.num_object * self.C.loader.frame_sample_len - x.shape[0]
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, x.shape[1], dtype=torch.float32)
            x = torch.cat([x, pad_tensor], dim=0)
        edge_index = stgraph['edge_index']
        edge_attr = torch.tensor(stgraph['edge_attr'].todense(), dtype=torch.float32)
        
        if self.transform_caption:
            r2l_caption = self.transform_caption(r2l_caption)
            l2r_caption = self.transform_caption(l2r_caption)


        if self.feature_mode == 'grid-obj-rel':
            object_video_feats = self.object_video_feats[vid]
            rel_feats = self.rel_feats[vid]

            if self.transform_frame:
                object_video_feats = self.transform_frame(object_video_feats).float()
                rel_feats = self.transform_frame(rel_feats).float()
            object_video_feats = object_video_feats.squeeze(0)
            rel_feats = rel_feats.squeeze(0)
            return vid, video_mask, x, edge_index, edge_attr, object_video_feats, rel_feats, r2l_caption, l2r_caption

        elif self.feature_mode == 'grid-rel':
            rel_feats = self.rel_feats[vid]
            if self.transform_frame:
                rel_feats = self.transform_frame(rel_feats).float()
            rel_feats = rel_feats.squeeze(0)
            return vid, video_mask, x, edge_index, edge_attr, rel_feats, r2l_caption, l2r_caption

        elif self.feature_mode == 'grid':
            return vid, video_mask, x, edge_index, edge_attr, r2l_caption, l2r_caption
        
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))

    def load_object_feats(self, frames, fin_o, fin_b, vid): 
        feats_b = fin_b[vid][()]
        feats_o = fin_o[vid][()]
        feats = np.concatenate((feats_b, feats_o), axis=1)
        num_paddings = frames - len(feats)
        if feats.size == 0:
            feats = np.zeros((frames, 1028)) 
        else:
            feats = feats.tolist() + [np.zeros_like(feats[0])
                                      for _ in range(num_paddings)]
        feats = np.asarray(feats)
        sampled_idxs = np.linspace(0, len(feats) - 1, frames, dtype=int) 
        feats = feats[sampled_idxs]
        assert len(feats) == frames
        return feats

    def load_gor_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        for i, model in enumerate(models):
            frames = self.C.loader.frame_sample_len
            if i == 0:
                continue

            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_{model}",
                                                                    self.phase, 'hdf5')
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_BFeat",
                                                                    self.phase, 'hdf5')

            with (h5py.File(fpath, 'r')) as fin, \
                (h5py.File(fpath_b, 'r')) as fin_b:

                data = fin
                vids = list(data.keys())
                for vid in vids:
                    feats = data[vid]
                    if i == 0:
                        continue

                    # Xử lý i == 1 (object_video_feats)
                    elif i == 1:
                        obj_feats = self.load_object_feats(frames=frames, fin_o=data, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(obj_feats)

                    # Xử lý i == 2 (rel_feats) - padding và lấy mẫu
                    elif i == 2:
                        if feats.size == 0 or len(feats) < frames:
                            num_paddings = frames - len(feats)
                            feats = np.zeros((frames, 1024)) if feats.size == 0 else np.vstack([feats] + [np.zeros_like(feats[0])] * num_paddings)

                        sampled_idxs = np.linspace(0, len(feats) - 1, frames, dtype=int)
                        self.rel_feats[vid].append(feats[sampled_idxs])
                    elif i == 3:
                        pad_len = frames - feats.shape[1]
                        if pad_len > 0:
                            zero_arr = torch.zeros([pad_len], dtype=torch.float32)
                            padded_feat = torch.cat([torch.tensor(feats[0]), zero_arr], dim=0)
                        else:
                            padded_feat = torch.tensor(feats[0])
                        self.video_mask[vid].append(padded_feat)
          
    def load_gr_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        for i, model in enumerate(models):
            frames = self.C.loader.frame_sample_len
            if i == 0: #grid
                continue

            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_{model}",
                                                                    self.phase, 'hdf5')
            with (h5py.File(fpath, 'r')) as fin:
                data = fin
                vids = list(data.keys())
                for vid in vids:
                    feats = data[vid]
                    if i == 0: #grid
                        continue
                    # Xử lý i == 1 (rel_feats)
                    elif i == 1:
                        if feats.size == 0 or len(feats) < frames:
                            num_paddings = frames - len(feats)
                            feats = np.zeros((frames, 1024)) if feats.size == 0 else np.vstack([feats] + [np.zeros_like(feats[0])] * num_paddings)

                        sampled_idxs = np.linspace(0, len(feats) - 1, frames, dtype=int)
                        self.rel_feats[vid].append(feats[sampled_idxs])
                    elif i == 2: #video_mask
                        pad_len = frames - feats.shape[1]
                        if pad_len > 0:
                            zero_arr = torch.zeros([pad_len], dtype=torch.float32)
                            padded_feat = torch.cat([torch.tensor(feats[0]), zero_arr], dim=0)
                        else:
                            padded_feat = torch.tensor(feats[0])
                        self.video_mask[vid].append(padded_feat)

    def load_g_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        print('\nEnter the load4 method. data_loader_fusion.py--row279')

        for i, model in enumerate(models):
            frames = self.C.loader.frame_sample_len
            if i == 0:
                continue
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_{model}",
                                                                    self.phase, 'hdf5')
            with (h5py.File(fpath, 'r')) as fin:

                data = fin
                vids = list(data.keys())
                for vid in vids:
                    feats = data[vid]
                    if i == 0:
                        continue
                    # Xử lý i == 1 (object_video_feats)
                    elif i == 1:
                        pad_len = frames - feats.shape[1]
                        if pad_len > 0:
                            zero_arr = torch.zeros([pad_len], dtype=torch.float32)
                            padded_feat = torch.cat([torch.tensor(feats[0]), zero_arr], dim=0)
                        else:
                            padded_feat = torch.tensor(feats[0])
                        self.video_mask[vid].append(padded_feat)


    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build_video_caption_pairs(self):
        self.load_captions()

        if self.feature_mode == 'grid-obj-rel':
            self.load_gor_feats()
            for vid in self.video_mask.keys():
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, r2l_caption, l2r_caption))

        elif self.feature_mode == 'grid-rel':
            self.load_gr_video_feats()
            for vid in self.video_mask.keys():
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, r2l_caption, l2r_caption))

        elif self.feature_mode == 'grid':
            self.load_g_video_feats()
            for vid in self.video_mask.keys():
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, r2l_caption, l2r_caption))
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))



class Corpus(object):
    """ Data Loader """
    def __init__(self, C, vocab_cls=CustomVocab, dataset_cls=CustomDataset, do_train=True):
        self.C = C
        self.feature_mode = C.feat.feature_mode
        self.do_train = do_train

        self.vocab = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        
        self.CustomVocab = vocab_cls
        self.CustomDataset = dataset_cls

        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(self.C.corpus),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),
        ])

        self.build()

    def build(self):
        self.build_vocab()
        self.build_data_loaders()

        del self.CustomVocab
        del self.CustomDataset
        gc.collect()

    def build_vocab(self):
        self.vocab = self.CustomVocab(
            # self.C.loader.total_caption_fpath,
            self.C.loader.train_caption_fpath,
            self.C.vocab.init_word2idx,
            self.C.loader.min_count,
            transform=self.transform_sentence)

    def build_data_loaders(self):
        """ Transformation """
        if self.C.loader.frame_sampling_method == "uniform":
            Sample = UniformSample
        elif self.C.loader.frame_sampling_method == "random":
            Sample = RandomSample
        else:
            raise NotImplementedError("Unknown frame sampling method: {}".format(self.C.loader.frame_sampling_method))

        self.transform_frame = transforms.Compose([
            Sample(self.C.loader.frame_sample_len),
            ToTensor(torch.float),
        ])
        self.transform_caption = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<S>']),
            PadLast(self.vocab.word2idx['<S>']),
            PadToLength(self.vocab.word2idx['<PAD>'], self.vocab.max_sentence_len + 2),  # +2 for <SOS> and <EOS>
            ToTensor(torch.long),
        ])
        
        if self.do_train == True: 
            self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
            self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)

            self.train_data_loader = self.build_data_loader(self.train_dataset, phase='train')
            self.val_data_loader = self.build_data_loader(self.val_dataset, phase='val')
        else: 
            self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)
            self.test_data_loader = self.build_data_loader(self.test_dataset, phase='test')
        
    def build_dataset(self, phase, caption_fpath):
        dataset = self.CustomDataset(self.C, phase, caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption,
        )
        return dataset
    
    def build_data_loader(self, dataset, phase):
        if phase == 'train' and torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            data_loader = DataLoader(
                dataset,
                batch_size=self.C.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.C.loader.num_workers,
                pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=1
            )
            return data_loader

        elif phase == 'val' or phase == 'test':
            sampler = SequentialSampler(dataset)
            data_loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                sampler=sampler,
                num_workers=self.C.loader.num_workers,
                pin_memory=True
            )
            return data_loader
