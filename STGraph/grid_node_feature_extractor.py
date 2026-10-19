import os
import torch
from tqdm import tqdm
import glob
import pathlib
import h5py
import numpy as np
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root



class args:
    msvd = False # or msvd = False for MSR-VTT
    slice_framepos = 2
    dset = "../"
    max_frames = 20
    eval_frame_order = 0 
    output_dir = 'pretrained'
    cache_dir = ''
    features_path = '..'
    msrvtt_csv = 'msrvtt.csv'
    max_words =32
    feature_framerate = 1
    cross_model = "cross-base"
    local_rank = 0
    pretrained_clip_name = "ViT-B/16" # Change to "ViT-B/32 if you use pretrained ViT with path size 32 in CLIP4Clip"
    
# MSVD
if args.msvd:
    dset_path = os.path.join(os.path.join(args.dset,'Dataset'),'MSVD')
    features_path = os.path.join(dset_path,'raw') # Raw uncompressed videos .avi    
    name_list = glob.glob(features_path+os.sep+'*')
    args.features_path = features_path

    url2id = {}
    data_path = os.path.join(os.path.join(dset_path,'captions','youtube-mapping.txt'))
    args.data_path = data_path
    for line in open(data_path,'r').readlines():
        url2id[line.strip().split(' ')[0]] = line.strip().split(' ')[-1]

    path_to_saved_models = "extracted/msvd_20_1fps"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
    save_file = path_to_saved_models+'/grid_node_features.hdf5'
    video_mask_file = path_to_saved_models+'/vid_mask.hdf5'
    args.max_words =30
    
    from dataloaders.dataloader_msvd_patch import MSVD_Loader
    
    args.feature_framerate = 1
    videos= MSVD_Loader(
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        transform_type = 0,
        patch=3,
        overlapped=0.5
    ) 
    
# MSRVTT
else:
    dset_path = os.path.join(os.path.join(args.dset,'Dataset'),'MSRVTT')
    features_path = os.path.join(dset_path,'videos')
    args.features_path = os.path.join(features_path,'all')
    data_path=os.path.join(dset_path,'MSRVTT_data.json')
    args.data_path = data_path
    args.msrvtt_csv = os.path.join(dset_path,'msrvtt.csv')
    name_list = glob.glob(args.features_path+os.sep+'*')

    path_to_saved_models = "extracted/msrvtt_grid"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
    save_file = path_to_saved_models+'/grid_node_features.hdf5'
    video_mask_file = path_to_saved_models+'/video_mask.hdf5'
    args.max_words = 73
    args.feature_framerate = 1
    from dataloaders.dataloader_msrvtt_patch import MSRVTT_RawDataLoader
    
    videos= MSRVTT_RawDataLoader(
        csv_path=args.msrvtt_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        transform_type = 0,
        patch=3,
        overlapped=0.5
    )
    
if args.msvd:
    model_file = os.path.join(ROOT, 'CLIP4Clip/ckpts/ckpt_msvd_retrieval_looseType/pytorch_model.bin.4')
else:
    model_file = os.path.join(ROOT, 'CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.4')
    
model_state_dict = torch.load(model_file, map_location='cpu')
cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
device = torch.device('cuda:0')
clip = model.clip.to(device)

# Number of desired pathces in the frame grid
NUM_PATCHES = 9
# Generate node features using CLIP4Clip to extract frame representation
with h5py.File(save_file, 'w') as f, h5py.File(video_mask_file, 'w') as f_mask:
    for i in tqdm(range(len(videos))):
        video_id, video_patches, video_mask = videos[i]
        
        f_mask.create_dataset(video_id, data = video_mask)
        length_frames = video_patches.shape[2]
        outputs = []
        for p in range(len(video_patches)):
            video=video_patches[p]
            tensor = video[0]
            tensor = tensor[video_mask[0]==1,:]
            tensor = torch.as_tensor(tensor).float()
            video_frame,num,channel,h,w = tensor.shape
            tensor = tensor.view(video_frame*num, channel, h, w)

            video_frame,channel,h,w = tensor.shape

            output = clip.encode_image(tensor.to(device), video_frame=video_frame).float().to(device)
            output = output.detach().cpu().numpy()
            outputs.append(output)
        outputs = np.stack(outputs)
        for o in range(len(video_mask[0])): # Iterate over frames
            if o < outputs.shape[1]:
                os = outputs[:, o, :]
            else:
                os = np.zeros((NUM_PATCHES,512)) # 512 is dimension of the extracted features
            f.create_dataset(video_id+'-'+str(o), data = os)
