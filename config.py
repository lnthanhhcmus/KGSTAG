import os
import time

# Repository root. Override with the environment variable KGSTAG_ROOT if the data live elsewhere.
ROOT = os.environ.get("KGSTAG_ROOT", os.path.dirname(os.path.abspath(__file__)))

class FeatureConfig(object):
    def __init__(self, model):
        self.model = model
        self.num_boxes = 60
        self.three_turple = 60

        if model in ['MSVD_GBased+OFeat+rel+videomask', 'MSR-VTT_GBased+OFeat+rel+videomask']:
            self.size = [1028, 300]
            self.feature_mode = 'grid-obj-rel'
        elif model in ['MSVD_GBased+rel+videomask', 'MSR-VTT_GBased+rel+videomask']:
            self.size = [512]
            self.feature_mode = 'grid-rel'
        elif model in ['MSVD_GBased+videomask', 'MSR-VTT_GBased+videomask']:
            self.size = [512]
            self.feature_mode = 'grid'
        else:
            raise NotImplementedError(f"Unknown model: {model}")

class VocabConfig(object):
    init_word2idx = {'<PAD>': 0, '<S>': 1}
    embedding_size = 512

class MSVDLoaderConfig(object):
    n_train = 1200
    n_val = 100
    n_test = 670
    
    total_caption_fpath = os.path.join(ROOT, "data/MSVD/metadata/MSR Video Description Corpus.csv")
    train_caption_fpath = os.path.join(ROOT, "data/MSVD/metadata/train.csv")
    val_caption_fpath = os.path.join(ROOT, "data/MSVD/metadata/val.csv")
    test_caption_fpath = os.path.join(ROOT, "data/MSVD/metadata/test.csv")
    min_count = 3
    max_caption_len = 20

    total_video_feat_fpath_tpl = os.path.join(ROOT, "data/{}/features/{}.{}")
    phase_video_feat_fpath_tpl = os.path.join(ROOT, "data/{}/features/{}_{}.{}")
    frame_sampling_method = 'uniform'
    frame_sample_len = 20
    num_workers = 4


class MSRVTTLoaderConfig(object):
    n_train = 6513
    n_val = 497
    n_test = 2990

    total_caption_fpath = os.path.join(ROOT, "data/MSR-VTT/metadata/total.json")
    train_caption_fpath = os.path.join(ROOT, "data/MSR-VTT/metadata/train.json")
    val_caption_fpath = os.path.join(ROOT, "data/MSR-VTT/metadata/val.json")
    test_caption_fpath = os.path.join(ROOT, "data/MSR-VTT/metadata/test.json")
    min_count = 3
    max_caption_len = 20

    total_video_feat_fpath_tpl = os.path.join(ROOT, "data/{}/features/{}.{}")
    phase_video_feat_fpath_tpl = os.path.join(ROOT, "data/{}/features/{}_{}.{}")
    frame_sampling_method = 'uniform'
    frame_sample_len = 20
    num_workers = 4


class TransformerConfig(object):
    d_model = 640
    n_heads = 10
    
    d_ff = 2048
    n_layers = 3
    dropout = 0.1
    n_heads_small = 12
    n_heads_big = 128

    max_frames = 50
    num_object = 9
    visual_num_hidden_layers = 3
    d_graph = 1024
    video_dim = 1024
    node_feat_dim = 512
    edge_dim = 1024
    visual_model = "visual-base"
    init_model = "./model/weight/univl.pretrained.bin"
    gnn_model_type = "transformer"
    cache_dir = ""
    local_rank = 0
    project_edge_dim = None
    no_skip = False
    last_average = False
    no_beta_transformer = False
    select_num = 0

class TrainConfig(object):
    def __init__(self, model_name, n_gpus):
        self.n_gpus = n_gpus
        self.feat = FeatureConfig(model_name)
        self.attention_mode = 1

        self.vocab = VocabConfig
        self.corpus = self.feat.model.split('_')[0]

        self.loader = {
            'MSVD': MSVDLoaderConfig,
            'MSR-VTT': MSRVTTLoaderConfig
        }[self.corpus]

        self.transformer = TransformerConfig
        self.transformer.max_frames = self.loader.frame_sample_len

        """ Optimization """
        self.epochs = {
            'MSVD': 30, 
            'MSR-VTT':  25,
        }[self.corpus]
        self.batch_size = 32
        self.optimizer = "AdamW"
        self.gradient_clip = 5.0
        self.lr = 1e-4

        self.gradient_accumulation_steps = {
            'MSVD': int(2 / self.n_gpus) , # if run on 1 gpu, effective batch size = 64
            'MSR-VTT':  int(4 / self.n_gpus), # if run on 1 gpu, effective batch size = 128
        }[self.corpus]
        
        self.weight_decay = 0.5e-5
        self.reg_lambda = 0.6
        self.beam_size = 5
        self.label_smoothing = 0.15

        """ Pretrained Model """
        self.pretrained_decoder_fpath = None

        """ Evaluate """
        self.metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

        """ ID """
        self.exp_id = "Transformer"
        self.feat_id = f"FEAT {self.feat.model} fsl-{self.loader.frame_sample_len} mcl-{self.loader.max_caption_len}"
        self.embedding_id = f"EMB {self.vocab.embedding_size}"
        self.transformer_id = f"Transformer d-{self.transformer.d_model}-N-{self.transformer.n_layers}-h-{self.transformer.n_heads}-h_big-{self.transformer.n_heads_big}-dp-{self.transformer.dropout}-sn-{self.transformer.select_num}"
        self.optimizer_id = f"OPTIM {self.optimizer} lr-{self.lr}-gac{self.gradient_accumulation_steps}-wd-{self.weight_decay}-rg-{self.reg_lambda}"
        self.hyperparams_id = f"bs-{self.batch_size}"
        if self.gradient_clip is not None:
            self.hyperparams_id += f" gc-{self.gradient_clip}"

        self.timestamp = time.strftime("%Y-%m-%d %X", time.localtime(time.time()))
        self.model_id = f"{self.feat.model} | {self.timestamp}"
        self.seed = 904666            # overwritten by --seed
        self.legacy_batching = False  # overwritten by --legacy_batching (see utils.graph_batch_offset)

        """ Log """
        self.path = ROOT
        self.log_dpath = os.path.join(self.path, f"logs/{self.corpus}/{self.model_id}")
        self.ckpt_dpath = os.path.join(self.path, f"checkpoints/{self.corpus}/{self.model_id}")
        self.captioning_dpath = os.path.join(self.path, f"captioning/{self.corpus}/{self.model_id}")
        self.ckpt_fpath_tpl = os.path.join(self.ckpt_dpath, "{}.ckpt")
        self.captioning_fpath_tpl = os.path.join(self.captioning_dpath, "{}.csv")
        
        self.save_from = 1
        self.save_every = 1

        """ TensorboardX """
        self.tx_train_loss = "loss/train"
        self.tx_train_r2l_cross_entropy_loss = "loss/train/r2l_loss"
        self.tx_train_l2r_cross_entropy_loss = "loss/train/l2r_loss"
        self.tx_val_loss = "loss/val"
        self.tx_val_r2l_cross_entropy_loss = "loss/val/r2l_loss"
        self.tx_val_l2r_cross_entropy_loss = "loss/val/l2r_loss"
        self.tx_lr = "params/vc_model_LR"