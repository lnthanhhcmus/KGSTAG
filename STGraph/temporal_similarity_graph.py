import os
import h5py
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root



NUM_NODES = 9
def cossim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


NODE_FEATURES, SAVE_FILE = None, None
msvd = False

if msvd: 
    NODE_FEATURES = os.path.join(ROOT, 'STGraph/extracted/msvd_object/object_node_features.hdf5') # can be object-based or grid-based
    SAVE_FILE = os.path.join(ROOT, "STGraph/extracted/msvd_object/object_temporal_graph.hdf5")
    print(SAVE_FILE)
    with h5py.File(NODE_FEATURES, 'r') as fp, h5py.File(SAVE_FILE, 'w') as f:
        # loop through all MSVD video
        for vidid in tqdm(range(1, 1971)):
            video_id = 'vid'+str(vidid)
            for frid in range(20):
                curr_node_feat = fp['vid'+str(vidid)+'-'+str(frid)][:]
                if frid == 0:
                    prev_node_feat = curr_node_feat
                    continue
                # create temp zero tensor of num_nodes
                Gt_temp = [[0.0] * NUM_NODES for i in range(NUM_NODES)]
                
                for k in range(len(prev_node_feat)):
                    for l in range(len(curr_node_feat)):
                        if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[l])==0):
                            continue
                            
                        # calculate the similarity between previous node and current node
                        Gt_temp[k][l] = math.exp(cossim(prev_node_feat[k], curr_node_feat[l]))
                        if np.isnan(Gt_temp[k][l]):
                            print(prev_node_feat[k])
                            print(curr_node_feat[l])
                            
                Gt = [[0.0] * NUM_NODES for i in range(NUM_NODES)]
                for k in range(len(prev_node_feat)):
                    for l in range(len(curr_node_feat)):
                        if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[l])==0):
                            continue
                        Gt[k][l] = Gt_temp[k][l]/sum(Gt_temp[k])
                    
                
                prev_node_feat = curr_node_feat
                f.create_dataset(video_id+'-'+str(frid), data = Gt)   
            
else:
    NODE_FEATURES = os.path.join(ROOT, 'STGraph/extracted/msrvtt_grid/grid_node_features.hdf5') # can be object-based or grid-based
    SAVE_FILE = os.path.join(ROOT, "STGraph/extracted/msrvtt_grid/grid_temporal_graph.hdf5")
    with h5py.File(NODE_FEATURES, 'r') as fp, h5py.File(SAVE_FILE, 'w') as f:
        # loop through all MSR-VTT video
        for vidid in tqdm(range(0, 10000)):
            video_id = 'video'+str(vidid)
            for frid in range(20):
                curr_node_feat = fp['video'+str(vidid)+'-'+str(frid)][:]
                if frid == 0:
                    prev_node_feat = curr_node_feat
                    continue
                # create temp zero tensor of num_nodes   
                Gt_temp = [[0.0] * NUM_NODES for i in range(NUM_NODES)]
                
                for k in range(len(prev_node_feat)):
                    for l in range(len(curr_node_feat)):
                        # if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[k])==0):
                        if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[l])==0):
                            continue
                            
                        # calculate the similarity between previous node and current node
                        Gt_temp[k][l] = math.exp(cossim(prev_node_feat[k], curr_node_feat[l]))
                        if np.isnan(Gt_temp[k][l]):
                            print(prev_node_feat[k])
                            print(curr_node_feat[l])
                            
                Gt = [[0.0] * NUM_NODES for i in range(NUM_NODES)]
                for k in range(len(prev_node_feat)):
                    for l in range(len(curr_node_feat)):
                        # if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[k])==0):
                        if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[l])==0):
                            continue
                        Gt[k][l] = Gt_temp[k][l]/sum(Gt_temp[k])
                    
                
                prev_node_feat = curr_node_feat
                f.create_dataset(video_id+'-'+str(frid), data = Gt)