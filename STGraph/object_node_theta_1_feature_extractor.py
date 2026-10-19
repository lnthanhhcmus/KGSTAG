import os
import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys
import glob
import json
import h5py
import math
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.ops.roi_align as roi_align
import pathlib
import torchvision.transforms as T
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root


device = torch.device('cuda')

# Arguments
class args:
    msvd = False # or msvd = False for MSR-VTT
    slice_framepos=2
    dset =ROOT # change based on dataset location
    max_frames = 20
    eval_frame_order = 0 
    output_dir='pretrained'
    cache_dir=''
    features_path='..'
    msrvtt_csv ='msrvtt.csv'
    data_path ='MSRVTT_data.json'
    max_words=32
    feature_framerate=1
    cross_model="cross-base"
    local_rank=0
    
# Load object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True, trust_repo=True)
model = model.to(device)
model.eval()
print()

# Set configuration
if args.msvd:

    dset_path = os.path.join(os.path.join(args.dset,'Dataset'),'MSVD')
    features_path = os.path.join(dset_path,'raw') # video .avi    
    name_list = glob.glob(features_path+os.sep+'*')
    args.features_path = features_path

    url2id = {}
    data_path =os.path.join(os.path.join(dset_path,'captions','youtube-mapping.txt'))
    args.data_path = data_path
    for line in open(data_path,'r').readlines():
        url2id[line.strip().split(' ')[0]] = line.strip().split(' ')[-1]

    path_to_saved_models = "extracted/msvd_object"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
    save_file = path_to_saved_models+'/MSVD_OBJECT_FEAT_FASTERRCNN_RESNET50.hdf5'
    args.max_words =30
    
else:
  
    dset_path = os.path.join(os.path.join(args.dset,'Dataset'),'MSRVTT')
    features_path = os.path.join(dset_path,'videos')
    args.features_path = os.path.join(features_path,'all')
    data_path=os.path.join(dset_path,'MSRVTT_data.json')
    args.data_path = data_path
    args.msrvtt_csv = os.path.join(dset_path,'msrvtt.csv')
    name_list = glob.glob(args.features_path+os.sep+'*')
    
    path_to_saved_models = "extracted/msrvtt"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
    save_file = path_to_saved_models+'/msrvtt_object_node_features.hdf5'
    args.max_words =73
    
# Feature extractor
def save_features(mod, inp, outp):
    features.append(outp)

# layer_to_hook = 'backbone.body.layer4.2.relu'
# layer_to_hook = 'roi_heads.box_roi_pool'

layer_to_hook = 'model.11.cv2.act'
# layer_to_hook = 'backbone.body.layer4'
for name, layer in model.model.model.named_modules():
# for name, layer in model.named_modules():
    if name == layer_to_hook:
        layer.register_forward_hook(save_features)
        
# Load dataset
if args.msvd :
    from dataloaders.dataloader_msvd import MSVD_Loader
    videos= MSVD_Loader(
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        transform_type = 1,
        data_path = args.data_path
) 
else:
    from dataloaders.dataloader_msrvtt import MSRVTT_RawDataLoader
    videos= MSRVTT_RawDataLoader(
        csv_path=args.msrvtt_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        transform_type = 1,
)
    
output_features = []
threshold = 0.5
model.conf = 0.5
features = None
stop = False
list_videoid = []

with torch.no_grad():
    with h5py.File(save_file, 'w') as f:
        for video_id,video,video_mask in tqdm(videos):
            print(video_id)
            if features is not None:
                del features
            features = []
            if (type(video) == bool):
                stop = True
            if stop:
                break

            tensor = video[0]

            roi_align_out_per_video = []
            for i in range(len(tensor)): 
                input = torch.tensor(tensor[i:i+1]).float()
                video_frame,num,channel,h,w = input.shape
                input = input.view(video_frame,channel, h, w)

                transform = T.ToPILImage()
                img = transform(input[0])

                output = model(img)

                spat_scale = min(features[i].shape[2]/input.shape[2], features[i].shape[3]/input.shape[3])
                roi_align_out_per_frame = []
                for j, box in enumerate(output.xyxy[0].cpu().numpy()): # for each box
                    if len(roi_align_out_per_frame)==9: # max object per frame is 9
                        break
                    roi_align_out = roi_align(features[i], [output.xyxy[0][:,:4][j:j+1]], output_size=1, spatial_scale=spat_scale, aligned=True)
                    roi_align_out_per_frame.append(torch.squeeze(roi_align_out).cpu().numpy())
                if len(roi_align_out_per_frame)<9: # add zero padding if less than 5 object
                    
                    for y in range(len(roi_align_out_per_frame), 9):
                        zero_padding = [0]*1024 # length of the roi_align_out is also 1024, hardcoded for now
                        roi_align_out_per_frame.append(zero_padding)
                
                roi_align_out_per_frame = np.stack(roi_align_out_per_frame)
                f.create_dataset(video_id+'-'+str(i), data = roi_align_out_per_frame)
                del output