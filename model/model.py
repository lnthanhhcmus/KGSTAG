import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import TransformerConv
from .modules.module_visual import VisualModel, VisualConfig
from .modules.until_module import LayerNorm
from collections import namedtuple
import logging
logger = logging.getLogger(__name__)


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def pad_mask(src, r2l_trg, trg, pad_idx, video_mask):
    if not isinstance(src, tuple): 
        src_vid_mask = (video_mask != pad_idx).unsqueeze(1)
        enc_src_mask = src_vid_mask
        dec_src_mask = src_vid_mask
        src_mask = (enc_src_mask, dec_src_mask)  
    elif len(src) == 3:
        src_vid_mask = (video_mask != pad_idx).unsqueeze(1)
        src_object_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
        src_rel_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_vid_mask, src_object_mask, src_rel_mask)
        dec_src_mask = src_vid_mask
        src_mask = (enc_src_mask, dec_src_mask)
    elif len(src) == 2:
        src_vid_mask = (video_mask != pad_idx).unsqueeze(1)
        src_rel_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_vid_mask, src_rel_mask)
        dec_src_mask = src_vid_mask
        src_mask = (enc_src_mask, dec_src_mask)        


    if trg is not None:
        if isinstance(src_mask, tuple):
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_vid_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_vid_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_vid_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
        else:
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask

    else:
        return src_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).cuda()

def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn
  
class FeatEmbedding(nn.Module):
    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)

class NormalizeVideo(nn.Module):
    def __init__(self, video_dim):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video
        
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module): # adjust max_len
    def __init__(self, dim, dropout, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)
 

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.LayerNorm = LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.LayerNorm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.LayerNorm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.LayerNorm(x + sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class EncoderLayerNoAttention(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayerNoAttention, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        return self.sublayer_connection[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))

        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x


class R2L_Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        return x


class L2R_Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
    
class TransC(nn.Module):
    def __init__(self, node_feat_dim, d_model, edge_dim, heads=4, project_edge_dim=None, more_skip=True, last_average=False, beta=True):
        super().__init__()
        self.lp = nn.Linear(node_feat_dim, d_model)
        self.more_skip = more_skip
        self.project_edge_dim = project_edge_dim
        if self.project_edge_dim is not None:
            self.lp_edge_attr = nn.Linear(edge_dim, project_edge_dim)
            edge_dim = project_edge_dim
        
        self.conv1 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)
        
        self.conv2 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)
        
        if last_average:
            self.conv3 = TransformerConv(d_model, d_model, heads, concat=False, edge_dim=edge_dim, aggr='mean', beta=beta)
        else:
            self.conv3 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)

    def forward(self, data):
        x = self.lp(data.x)
        if self.project_edge_dim is not None:
            e = F.relu(self.lp_edge_attr(data.edge_attr))
        else:
            e = data.edge_attr
        if self.more_skip:
            x = F.relu(x + self.conv1(x, data.edge_index, e))
            x = F.relu(x + self.conv2(x, data.edge_index, e))
            x = F.relu(x + self.conv3(x, data.edge_index, e))
        else:
            x = F.relu(self.conv1(x, data.edge_index, e))
            x = F.relu(self.conv2(x, data.edge_index, e))
            x = F.relu(self.conv3(x, data.edge_index, e))
        return x
    
class STGraphEncoder(nn.Module):
    def __init__(self, visual_model_name, heads_type, state_dict=None, cache_dir=None, type_vocab_size=2, task_config=None):
        super(STGraphEncoder, self).__init__()
        
        if task_config and not hasattr(task_config, "local_rank"):
            task_config.__dict__["local_rank"] = 0
        elif task_config and task_config.local_rank == -1:
            task_config.local_rank = 0
        self.task_config = task_config
        
        self.visual_config, _ = VisualConfig.get_config(
            visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        assert self.task_config.max_frames <= self.visual_config.max_position_embeddings
        self.visual_config = update_attr("visual_config", self.visual_config, "num_hidden_layers",
                                         self.task_config, "visual_num_hidden_layers")
        self.visual_config = update_attr("visual_config", self.visual_config, "num_attention_heads",
                                         self.task_config, heads_type)
        # self.visual_config = update_attr("visual_config", self.visual_config, "hidden_size",
        #                                  self.task_config, "hidden_size")
        # self.visual_config = update_attr("visual_config", self.visual_config, "vocab_size",
        #                                  self.task_config, "vocab_size")

        self.visual = VisualModel(self.visual_config)
        self.normalize_video = NormalizeVideo(task_config.video_dim)
        
        self.check = False
        if self.visual_config.hidden_size != task_config.d_model:
            self.check = True
            self.lp = nn.Linear(self.visual_config.hidden_size, task_config.d_model)

        # Áp dụng trọng số pretrain nếu có
        if state_dict is not None:
            self.init_preweight(state_dict)

        self.apply(self.init_weights)

    def init_preweight(self, state_dict, prefix=None):
        """ Load trọng số từ mô hình pretrain """
        old_keys, new_keys = [], []
        for key in state_dict.keys():
            new_key = key.replace('gamma', 'weight').replace('beta', 'bias') if 'gamma' in key or 'beta' in key else None
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            state_dict = {prefix + key: state_dict.pop(key) for key in list(state_dict.keys())}

        missing_keys, unexpected_keys, error_msgs = [], [], []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self, prefix='')

        if prefix is None and (self.task_config is None or self.task_config.local_rank == 0):
            logger.info("-" * 20)
            if missing_keys:
                logger.info(f"Weights of {self.__class__.__name__} not initialized from pretrained model:\n   " + "\n   ".join(missing_keys))
            if unexpected_keys:
                logger.info(f"Weights from pretrained model not used in {self.__class__.__name__}:\n   " + "\n   ".join(unexpected_keys))
            if error_msgs:
                logger.error(f"Weights from pretrained model cause errors in {self.__class__.__name__}:\n   " + "\n   ".join(error_msgs))

    def init_weights(self, module):
        """ Initialize the weights. """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.visual_config.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'beta') and hasattr(module, 'gamma'):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        
        if self.check: 
            return self.lp(visual_output)
        return visual_output
    
class GraphTransformer(nn.Module):
	def __init__(self, head_type, state_dict=None, cache_dir=None, args=None):
		super().__init__()
		self.transc = TransC(node_feat_dim=args.node_feat_dim, d_model=args.d_graph, edge_dim=args.edge_dim,
								project_edge_dim=args.project_edge_dim, more_skip=args.no_skip==False, last_average=args.last_average, beta=args.no_beta_transformer==False)
		self.avgPool = nn.AvgPool2d((args.num_object,1)) # number of patches or objects
		self.stg_encoder = STGraphEncoder(args.visual_model, head_type, cache_dir=cache_dir, state_dict=state_dict, task_config=args)
  
	def forward(self, geo_graph, video_mask=None, batch_size=None, n_node=None):
		fo_convolved = self.transc(geo_graph)
		fo_convolved = fo_convolved.unflatten(0, (batch_size, n_node))
		fo_convolved = self.avgPool(fo_convolved)
		visual_output = self.stg_encoder(fo_convolved, video_mask)
		return visual_output

class VCModel(nn.Module):
    def __init__(self, vocab, model_state_dict, cache_dir, feature_mode, C_tran, d_feat, attention_mode, device):
        super(VCModel, self).__init__()

        self.vocab = vocab
        self.device = device
        self.feature_mode = feature_mode
        self.attention_mode = attention_mode

        c = copy.deepcopy
        attn = MultiHeadAttention(C_tran.n_heads, C_tran.d_model, C_tran.dropout)
        feed_forward = PositionWiseFeedForward(C_tran.d_model, C_tran.d_ff)

        if self.feature_mode == 'grid-obj-rel':
            self.object_src_embed = FeatEmbedding(d_feat[0], C_tran.d_model, C_tran.dropout)
            self.rel_src_embed = FeatEmbedding(d_feat[1], C_tran.d_model, C_tran.dropout)
            self.encoder = Encoder(C_tran.n_layers, EncoderLayer(C_tran.d_model, c(attn), c(feed_forward), C_tran.dropout))
            self.encoder_no_attention = Encoder(C_tran.n_layers, EncoderLayerNoAttention(C_tran.d_model, c(attn), c(feed_forward), C_tran.dropout))
        elif self.feature_mode == 'grid-rel':
            self.rel_src_embed = FeatEmbedding(d_feat[0], C_tran.d_model, C_tran.dropout)
            if self.attention_mode == 1 or self.attention_mode == 2:
                self.encoder = Encoder(C_tran.n_layers, EncoderLayer(C_tran.d_model, c(attn), c(feed_forward), C_tran.dropout))
            elif self.attention_mode == 3:
                self.encoder_no_attention = Encoder(C_tran.n_layers, EncoderLayerNoAttention(C_tran.d_model, c(attn), c(feed_forward), C_tran.dropout))

        self.stg_encoder_big = GraphTransformer(head_type='n_heads_big', state_dict=model_state_dict, cache_dir=cache_dir, args=C_tran) # STE
        self.stg_encoder = GraphTransformer(head_type='n_heads_small', state_dict=model_state_dict, cache_dir=cache_dir, args=C_tran) # ORE
        self.trg_embed = TextEmbedding(vocab.n_vocabs, C_tran.d_model)
        self.pos_embed = PositionalEncoding(C_tran.d_model, C_tran.dropout) 
        
        self.r2l_decoder = R2L_Decoder(C_tran.n_layers, DecoderLayer(C_tran.d_model, c(attn), c(feed_forward), sublayer_num=3, dropout=C_tran.dropout))
        self.l2r_decoder = L2R_Decoder(C_tran.n_layers, DecoderLayer(C_tran.d_model, c(attn), c(feed_forward), sublayer_num=4, dropout=C_tran.dropout))
        self.generator = Generator(C_tran.d_model, vocab.n_vocabs)
        
    def encode(self, src, src_mask, feature_mode_two=False):
        if isinstance(src, (list, tuple)):
            batch = src_mask[0].shape[0]
            n_nodes = src[0].x.shape[0] // batch
        else:
            batch = src_mask.shape[0]
            n_nodes = src.x.shape[0] // batch
    
        if feature_mode_two:
            # VGE_B (memory of the backward decoder) sees the spatio-temporal graph only
            if isinstance(src, (list, tuple)):
                return self.stg_encoder_big(src[0], src_mask[0], batch, n_nodes)
            return self.stg_encoder_big(src, src_mask, batch, n_nodes)
    
        if self.feature_mode == 'grid-obj-rel':
            x1 = self.stg_encoder(src[0], src_mask[0], batch, n_nodes)
    
            x2 = self.object_src_embed(src[1])
            x2 = self.encoder(x2, src_mask[1])
    
            x3 = self.rel_src_embed(src[2])
            x3 = self.encoder_no_attention(x3, src_mask[2])
            return x1 + x2 + x3
    
        elif self.feature_mode == 'grid-rel':
            x1 = self.stg_encoder(src[0], src_mask[0], batch, n_nodes)
    
            x2 = self.rel_src_embed(src[1])
            if self.attention_mode == 1:
                x2 = self.encoder(x2, src_mask[1])
            elif self.attention_mode == 2:
                x2 = self.pos_embed(x2)
                x2 = self.encoder(x2, src_mask[1])
            elif self.attention_mode == 3:
                x2 = self.encoder_no_attention(x2, src_mask[1])
            return x1 + x2
    
        elif self.feature_mode == 'grid':
            return self.stg_encoder(src, src_mask, batch, n_nodes)
    
    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.trg_embed(r2l_trg)
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
    
    def forward(self, src, r2l_trg, trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        enc_src_mask, dec_src_mask = src_mask
        
        r2l_encoding_outputs = self.encode(src, enc_src_mask, feature_mode_two=True)
        encoding_outputs = self.encode(src, enc_src_mask)

        r2l_outputs = self.r2l_decode(r2l_trg, r2l_encoding_outputs, dec_src_mask, r2l_trg_mask)
        l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, r2l_outputs, r2l_pad_mask)
        

        r2l_pred = self.generator(r2l_outputs)
        l2r_pred = self.generator(l2r_outputs)
        return r2l_pred, l2r_pred
    
    def greedy_decode(self, batch_size, src_mask, memory, max_len):

        eos_idx = self.vocab.word2idx['<S>']
        r2l_hidden = None
        with torch.no_grad():
            output = torch.ones(batch_size, 1).fill_(eos_idx).long().cuda()
            for i in range(max_len + 2 - 1):
                trg_mask = subsequent_mask(output.size(1))
                dec_out = self.r2l_decode(output, memory, src_mask, trg_mask)  # batch, len, d_model
                r2l_hidden = dec_out
                pred = self.generator(dec_out)  # batch, len, n_vocabs
                next_word = pred[:, -1].max(dim=-1)[1].unsqueeze(1)  # pred[:, -1]([batch, n_vocabs])
                output = torch.cat([output, next_word], dim=-1)
        return r2l_hidden, output

    def r2l_beam_search_decode(self, batch_size, src, src_mask, model_encodings, beam_size, max_len):
        end_symbol = self.vocab.word2idx['<S>']
        start_symbol = self.vocab.word2idx['<S>']

        r2l_outputs = None

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        # model_encodings = self.encode(src, src_mask)

        # 1.2 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'grid':
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.x)).to(self.device))
            # elif self.feature_mode == 'two' or 'three' or 'four' or 'grid':
            else:
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[1].data)).to(self.device))
            r2l_outputs = out

            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        return r2l_outputs, completed_hypotheses

    def beam_search_decode(self, src, beam_size, max_len, video_mask):
        """
                An Implementation of Beam Search for the Transformer Model.
                Beam search is performed in a batched manner. Each example in a batch generates `beam_size` hypotheses.
                We return a list (len: batch_size) of list (len: beam_size) of Hypothesis, which contain our output decoded sentences
                and their scores.
                :param src: shape (sent_len, batch_size). Each val is 0 < val < len(vocab_dec). The input tokens to the decoder.
                :param max_len: the maximum length to decode
                :param beam_size: the beam size to use
                :return completed_hypotheses: A List of length batch_size, each containing a List of beam_size Hypothesis objects.
                    Hypothesis is a named Tuple, its first entry is "value" and is a List of strings which contains the translated word
                    (one string is one word token). The second entry is "score" and it is the log-prob score for this translated sentence.
                Note: Below I note "4 bt", "5 beam_size" as the shapes of objects. 4, 5 are default values. Actual values may differ.
                """
        # 1. Setup
        start_symbol = self.vocab.word2idx['<S>']
        end_symbol = self.vocab.word2idx['<S>']

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        src_mask = pad_mask(src, r2l_trg=None, trg=None, pad_idx=self.vocab.word2idx['<PAD>'], video_mask=video_mask)
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        if self.feature_mode == 'grid':
            batch_size = video_mask.shape[0]
            enc_src_mask = src_mask[0]
            dec_src_mask = src_mask[1]
            r2l_model_encodings = self.encode(src, enc_src_mask, feature_mode_two=True)
            model_encodings = self.encode(src, enc_src_mask)
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, dec_src_mask,
                                                                               model_encodings=r2l_model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)
        # elif self.feature_mode == 'two' or 'three' or 'four' or 'grid':
        else:
            batch_size = src[1].shape[0]
            enc_src_mask = src_mask[0]
            dec_src_mask = src_mask[1]
            r2l_model_encodings = self.encode(src, enc_src_mask, feature_mode_two=True)
            # model_encodings = r2l_model_encodings
            model_encodings = self.encode(src, enc_src_mask)

            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, dec_src_mask,
                                                                               model_encodings=r2l_model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)

        # 1.2 Setup r2l target output
        # r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, src_mask,
        #                                                                    model_encodings=model_encodings,
        #                                                                    beam_size=1, max_len=max_len)
        # r2l_memory, r2l_completed_hypotheses = self.greedy_decode(batch_size, src_mask, model_encodings, max_len)
        # beam_r2l_memory = [copy.deepcopy(r2l_memory) for _ in range(beam_size)]
        # 1.3 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l, r2l_memory_l = [], [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                if self.feature_mode == 'grid':
                    src_mask_l += [dec_src_mask[i:i + 1]] * cur_beam_size
                # elif self.feature_mode == 'two' or 'three' or 'four' or 'grid':
                else:     
                    src_mask_l += [dec_src_mask[i:i + 1]] * cur_beam_size
                r2l_memory_l += [r2l_memory[i: i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            r2l_memory_cur = torch.cat(r2l_memory_l, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'grid':
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.x)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            # elif self.feature_mode == 'two' or 'three' or 'four' or 'grid':
            else:
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[1].data)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print('completed_hypotheses', completed_hypotheses)
        return r2l_completed_hypotheses, completed_hypotheses
