import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import pathlib
import sparse
import pickle
import gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root


device = torch.device('cuda')

# Argument
class args:
    dataset = 'msrvtt_grid' # or dataset = 'msrvtt'    
    num_edge_feat = 1024 # Dimension of edge feature

# Output ST Graph path
path_to_saved_models = "extracted/"+args.dataset
pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)

# Load spatial and temporal features
try:
    # sf_file =path_to_saved_models+'/<Path to spatial graph>.hdf5'
    # ft_file =path_to_saved_models+'/<Path to temporal graph>.hdf5'
    
    # save_file = path_to_saved_models+'/<Desired filename>.hdf5'
    
    sf_file = os.path.join(ROOT, f"STGraph/extracted/{args.dataset}/grid_spatio_graph.hdf5")
    ft_file = os.path.join(ROOT, f"STGraph/extracted/{args.dataset}/grid_temporal_graph.hdf5")
    
    save_file = os.path.join(ROOT, f"STGraph/extracted/{args.dataset}/stg.hdf5")
    fs = h5py.File(sf_file,'r')
    ft = h5py.File(ft_file,'r')
except Exception as e:
    print(e)    
   
num_object = 9 # Number of patches
num_frame = 20 # Num of frames

with open(save_file, 'ab+') as handle:
    keys = list(fs.keys())
    grouped_keys = {}
    for key in keys:
        vid, frame_id = key.split('-')
        if vid not in grouped_keys:
            grouped_keys[vid] = []
        grouped_keys[vid].append(int(frame_id))
        
    for vid in tqdm(grouped_keys):
        frame_ids = sorted(grouped_keys[vid])
    
        sgraph_list = {}
        tgraph_list = {}
        
        for i, k_fr in enumerate(frame_ids):
            video_id = f"{vid}-{k_fr}"
            
            sgraph = fs[video_id][:]
            for k in range(num_object):
                for l in range(num_object):
                    if isinstance(sgraph[k][l], str):
                        sgraph[k][l] = eval(sgraph[k][l])
                    else:
                        sgraph[k][l] = sgraph[k][l].astype(np.float64)
            sgraph_list[k_fr] = sgraph          
            
            if video_id in ft:
                tgraph = ft[video_id][:]
                tgraph_list[k_fr] = np.concatenate((np.expand_dims(tgraph, axis=2), np.zeros((num_object, num_object, 1023))), axis=2) 
        
        mgraph = np.zeros((num_frame * num_object, num_frame * num_object, args.num_edge_feat))        
        for i, k_fr in enumerate(frame_ids):
            s_start = i*num_object
            s_end = (i*num_object)+num_object
            t_start = s_start+num_object
            t_end = s_start+(num_object*2)
            
            mgraph[s_start:s_end,s_start:s_end] = sgraph_list[k_fr]
            if i < len(frame_ids)-1:
                mgraph[s_start:s_end, t_start:t_end] =tgraph_list[frame_ids[i+1]]   
                
        print(vid)
        y = sparse.COO(mgraph)
        s = {vid: y}
        pickle.dump(s, handle)  
        
        del mgraph
        gc.collect()  