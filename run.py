from __future__ import print_function
import os
import gc
import torch
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from model.model import VCModel
from collections import OrderedDict
from utils import dict_to_cls, get_predicted_captions, get_groundtruth_captions, save_result, score
import  logging 

logger = logging.getLogger(__name__)

def build_loader(ckpt_fpath, do_train):
    checkpoint = torch.load(ckpt_fpath, weights_only=False)
    config = dict_to_cls(checkpoint['config'])
    """ Build Data Loader """
    if config.corpus == "MSVD":
        corpus = MSVD(config, do_train)
    elif config.corpus == "MSR-VTT":
        corpus = MSRVTT(config, do_train)
    else:
        raise "Error in build_loader"

    test_iter, vocab = corpus.test_data_loader, corpus.vocab
    
    r2l_test_vid2GTs, l2r_test_vid2GTs = get_groundtruth_captions(test_iter, vocab,
                                                                 config.feat.feature_mode)
    
    logger.info('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, config.loader.min_count))
    
    del corpus
    del  r2l_test_vid2GTs
    gc.collect()
    return test_iter, vocab, l2r_test_vid2GTs


def run(ckpt_fpath, test_iter, vocab, ckpt, l2r_test_vid2GTs, f, captioning_fpath, C, device):
    captioning_dpath = os.path.dirname(captioning_fpath)

    if not os.path.exists(captioning_dpath):
        os.makedirs(captioning_dpath)

    """ Load Config """
    checkpoint = torch.load(ckpt_fpath, weights_only=False)
    config = dict_to_cls(checkpoint['config'])

    """ Build Models """
    model = VCModel(vocab, None, None, C.feat.feature_mode, C.transformer, C.feat.size, C.attention_mode, device)
    model = model.to(device)

    state_dict = checkpoint['vc_model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)

    """ Test Set """
    logger.info('Finish the model load in CUDA. Try to enter Test Set.')
    r2l_test_vid2pred, l2r_test_vid2pred = get_predicted_captions(test_iter, model, config.beam_size, config.loader.max_caption_len, config.feat.feature_mode, device)
    l2r_test_scores = score(l2r_test_vid2pred, l2r_test_vid2GTs)
    logger.info("[TEST L2R] in {} is {}".format(ckpt, l2r_test_scores))

    f.write(ckpt + " result: ")
    f.write("[TEST L2R] in {} is {}".format(ckpt, l2r_test_scores))
    f.write('\n')

    save_result(l2r_test_vid2pred, l2r_test_vid2GTs, captioning_fpath)

    del checkpoint
    del model
    del r2l_test_vid2pred
    del l2r_test_vid2pred
    gc.collect()
    torch.cuda.empty_cache()