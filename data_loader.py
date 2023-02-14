import pandas as pd
import numpy as np
from utils import get_frames_data, get_edges_index
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

###############################################################
# do the matrix multiplication to obtain the graph matrix for
# each frame
###############################################################
def generate_graph_matrix(data, adj_m):
    print(data[0].shape, adj_m.shape)
    graph =  adj_m @ data[0]
    #graph = np.dot(data, adj_m)
    #print(graph[1])
    #result = np.dot(data[0],adj_m)
    #print(result)

    result = []
    for keypoints in data:
        result.append(np.dot(adj_m,keypoints))#adj_m @ keypoints)
    return np.stack(result)
    '''
    for pos,  instance in enumerate(data):
        print(pos)
        result.append(np.dot(data,adj_m))
    '''

###############################################################
# To get the Adjacent matrix
# from a excel that have joints conexions
###############################################################
def get_adjacent_matrix(keypoints_number=71):
    
    points_joints_info = pd.read_csv(f'./points_{keypoints_number}.csv')
    # we subtract one because the list is one start (we wanted it to start in zero)
    ori = points_joints_info.origin-1
    tar = points_joints_info.tarjet-1

    # Matrix size nxn
    n = keypoints_number

    # create the adj matrix filling vales with zeros
    A = np.zeros((n, n))

    for origin, target in zip(ori, tar):
 
        A[origin, target] = 1
        A[target, origin] = 1

    return A

###############################################################
# Reduce the data to only the points we wanted
###############################################################
def data_format(data, kp_keys):
   
    data = data[:,:,kp_keys]

    data = np.moveaxis(data,2,1)

    return data

###############################################################
# To get the keypoint position in the array of certain points
# that are defined in the landmarks_ref. There are already for
# 71 and 29 points
###############################################################
def get_keypoint_keys(points_joints_info, keypoints_model, keypoints_number=71):

    if keypoints_model == 'openpose':
        points = np.array(points_joints_info.op_pos)-1
    if keypoints_model == 'mediapipe':
        points = np.array(points_joints_info.mp_pos)-1
    if keypoints_model == 'wholepose':
        points = np.array(points_joints_info.wp_pos)-1

    points = list(points)

    return points


###############################################################
# Data instance of a dataset of keypoints of differente frames
###############################################################
class keypointFramesDataset(Dataset):
    def __init__(self, path, keypoints_model, num_points):

        data = get_frames_data(path)

        points_joints_info = pd.read_csv(f'./points_{num_points}.csv')

        kp_keys = get_keypoint_keys(points_joints_info,
                                    keypoints_model=keypoints_model, 
                                    keypoints_number=num_points)

        data = data_format(data, kp_keys)
        adj = get_adjacent_matrix()

    
        self.edges = adj
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_sample = self.data[index]
        edges = self.edges
        
        return data_sample, edges
