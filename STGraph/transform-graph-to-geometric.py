import os
import torch
import pickle
import itertools
import h5py
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
import pathlib
import numpy as np
import sparse
import gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root


# Arguments
class args:
    dataset = 'msrvtt_grid' # or dataset = 'msvd'
    size = 10000
    
# Set configuration
path_to_saved_models = "extracted/"+args.dataset
pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
stg_file = os.path.join(ROOT, f'STGraph/extracted/{args.dataset}/stg.hdf5')
fo_file = os.path.join(ROOT, f'STGraph/extracted/{args.dataset}/grid_node_features.hdf5')

def stack_node_features(pathfile):
    fo_input = h5py.File(pathfile, "r")
    fo_list = {}
    for i,key in tqdm(enumerate(fo_input.keys()), total=len(fo_input.keys())):
        a = key.split('-')

        if a[0] not in fo_list:
            fo_list[a[0]] = {}
        fo_list[a[0]][int(a[1])] = fo_input[key][:]

    fo_stacked = {}
    for key in fo_list.keys():
        stacked = []
        for k_fr in sorted(fo_list[key].keys()):
            stacked.append(fo_list[key][k_fr])
        fo_stacked[key] = np.vstack(stacked)
        
    return fo_stacked

def generate_graph_data(stg_vid, fo_vid):
    """Generate graph data for every vid_id STG & FO"""
    t =[]
    attr =[]
    n_rows = stg_vid.shape[0]
    n_columns = stg_vid.shape[1]
    n_dim_feature = stg_vid.shape[2]
    n_dim_fo = fo_vid.shape[1]
    
    allzero = False
    
    # Edge index
    edge_index = torch.tensor(list(map(list, itertools.product(np.arange(n_rows), repeat=2))), dtype=torch.long)
      
    # Edge feature
    edge_attr = torch.tensor(stg_vid.todense()[:n_rows, :n_columns], dtype=torch.float).reshape(n_rows * n_columns, n_dim_feature)

    for i in range (len(edge_attr)):
        allzero = torch.sum(edge_attr[i])
        if allzero > 0:
            t.append(edge_index[i])
            attr.append(edge_attr[i])

    # Node feature
    if(len(t)==0):
        v=edge_index[0].unsqueeze(0)
        attr = edge_attr[0].unsqueeze(0)
        allzero = True
    else:
        v = torch.stack(t)
        attr = torch.stack(attr)
   
    x = torch.tensor(fo_vid[:n_rows], dtype=torch.float)


    # Generate the graph
    data = Data(x=x, edge_index=v.t().contiguous(), edge_attr=attr)
    del attr
    del v
    del t
    
    return data, allzero


# Prepare action data
stg = [None for _ in range(args.size)]
fo = stack_node_features(fo_file)
cnt = 0  
with (open(stg_file, "rb")) as openfile:
    while True:
        try:
            data = pickle.load(openfile)
            if 'msvd' in args.dataset:
                key = int(list(data.keys())[0].split("vid")[1])
                stg[key - 1] = data
            else:
                key = int(list(data.keys())[0].split("video")[1])
                stg[key] = data
            
            cnt += 1
            if cnt==10000:
                break
        except EOFError:
            break
        
# Generate Pytorch geometric data
datas = {}
index=[]
for i in tqdm(range(len(stg))):
    if 'msvd' in args.dataset:
        id = 'vid' + str(i+1)
    else:
        id = 'video' + str(i)
        
    stg_vid = stg[i][id]
    fo_vid = fo[id]

    datas[id], allzero = generate_graph_data(stg_vid, fo_vid)
    if allzero:
        index.append(i)

# Save memory by deleting previous data (if the cell has been run multiple times)
stg = None
fo = None
contents = None
ids = None
del stg
del fo
del contents
del ids
gc.collect()


num_object = 9
num_edge_features = 1024
num_frame = 20
num_node = num_object * num_frame
max_ = 0

for key in datas.keys():
    datas[key].edge_attr = sparse.COO(np.array(datas[key].edge_attr))

for g in datas:
    max_ = max(datas[g].edge_index.shape[1], max_)
    
hmap = {}
for g in tqdm(datas):
    for i in range(datas[g].edge_index.shape[1]):
        key = str(g)+'-'+str(datas[g].edge_index[0][i].item())+'-'+str(datas[g].edge_index[1][i].item())
        hmap[key] = 1

# Generate the data structure
for g in tqdm(datas):
    curr_size = datas[g].edge_index.shape[1]
    
    if curr_size < max_:
        counter = max_ - curr_size
        done = False
        if type(datas[g].edge_attr)!=np.ndarray:
            datas[g].edge_attr = datas[g].edge_attr.todense()
        for i in range(num_node):
            for j in range(num_node):
                key = str(g)+str(i)+'-'+str(j)
                if (key in hmap) == False:
                    datas[g].edge_index = torch.hstack((datas[g].edge_index, torch.tensor([[i],[j]])))
                    datas[g].edge_attr = np.vstack((datas[g].edge_attr,np.zeros(num_edge_features)))
                    counter -= 1
                    
                    if counter==0:
                        done =True
                        break
            if done:
                break
        datas[g].edge_attr = sparse.COO(datas[g].edge_attr)
        
# Save
action_graph = path_to_saved_models+'/stg.pickle'
with open(action_graph, 'wb') as fp:
     pickle.dump(datas, fp)
        
print("SPATIO TEMPORAL ACTION GRAPH SUCCESSFULLY SAFE")