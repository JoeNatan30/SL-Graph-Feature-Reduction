import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import cv2

###############################################################
# Use csv keypoints information of certain num of points
# and return and array of origin-target conections
###############################################################
def get_edges_index(keypoints_number=71):
    
    points_joints_info = pd.read_csv(f'./points_{keypoints_number}.csv')
    # we subtract one because the list is one start (we wanted it to start in zero)
    ori = points_joints_info.origin-1
    tar = points_joints_info.tarjet-1

    ori = np.array(ori)
    tar = np.array(tar)

    return np.array([ori,tar])


###############################################################
# Read the data and convert it into list
###############################################################
def read_h5(path):

    classes = []
    videoName = []
    data = []

    #read file
    with h5py.File(path, "r") as f:
        for index in f.keys():
            classes.append(f[index]['label'][...].item().decode('utf-8'))
            videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
            data.append(f[index]["data"][...])
    
    return classes, videoName, data

###############################################################
# get the list of data from read_h5 and convert it into 
# Dataframe
###############################################################
def create_df(path):

    glosses, videoName, data = read_h5(path)

    df = pd.DataFrame.from_dict({
        "classes":glosses,
        "videoName": videoName,
        "data":data,  
    })
    return df

###############################################################
# obtain train or val data and reshape
# it to have a dataset of frames instead of videos
###############################################################
def get_frames_data(path):
   
    df = create_df(path)

    data = np.array(df.data)
    result = np.concatenate(data, axis=0)

    return result

###############################################################
# - Keypoint should be =>  [x, y] where x is an array of xs coords
# the same for y.
# - connections are the edges of the graph and should be [[a0,b0][a1,b1], ... , [an,bn]]
# the a and b value are the coneccion between points.
# 
# This function returns an image of the keypoints, if connections are used, it draws conections lines
###############################################################
def prepare_keypoints_image(keypoints, connections=[]):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}

    # DRAW POINTS
    img = np.zeros((256, 256, 3), np.uint8)

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)

        cv2.circle(img, (cor_x, cor_y), 1, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)

    # DRAW JOINT LINES
    for start_p, end_p in connections:
        if start_p in part_line and end_p in part_line:
            start_p = part_line[start_p]
            end_p = part_line[end_p]
            cv2.line(img, start_p, end_p, (0,255,0), 2)

    return img