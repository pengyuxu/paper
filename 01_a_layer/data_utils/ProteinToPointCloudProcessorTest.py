import numpy as np
import trimesh
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import time
import sys


import pandas as pd
#from processor_utils import get_mesh, is_cad_model, remove_extension
from sklearn import preprocessing


class DataProcessor(Dataset):
    def __init__(self, root_path, file_list, pc_folder = "/txt/", split = "test"):

        self.root_path = root_path
        self.pc_folder = pc_folder


        self.list_of_points = [] #[None] * len(self.datapath)
        #self.list_of_labels = category_list#[None] * len(self.datapath)
        self.list_of_names = file_list

        # labelled data are split across subfolders based on category
        # this can be unified with unlabelled data
        for file in self.list_of_names:
            #train set
            if split == "train":
                #samples = np.loadtxt(self.root_path + self.pc_folder + file + "_pc.txt", delimiter=",").astype(np.float32)
                samples=np.load(self.root_path + self.pc_folder + file + "_pc.npy").astype(np.float32)
            else:
                #samples=np.loadtxt(self.root_path + self.pc_folder + file[:-4] + "_pc.txt", delimiter=",").astype(np.float32)
                samples=np.load(self.root_path + self.pc_folder + file[:-4] + "_pc.npy").astype(np.float32)


            self.list_of_points.append(samples)


        self.list_of_points = np.array(self.list_of_points)
        #self.list_of_labels = np.array(self.list_of_labels)
        #self.list_of_labels = self.list_of_labels.astype(np.int32)


    def __len__(self):
        return self.list_of_points.shape[0]

    def __getitem__(self, index):
        #point_set, label = self.list_of_points[index], self.list_of_labels[index]
        return self.list_of_points[index]



if __name__ == '__main__':

    n_points = 2048
    root_path = ".../shrec2025/train_set_vtk/"
    data = pd.read_csv(".../shrec2025/train_set_2.csv")
    filenames = data["protein_id"]
    labels = data["class_id"]

    train_data = filenames[:7428]
    train_labels = labels[:7428]

    validation_data = filenames[7428:]
    validation_labels = labels[7428:]


    start = time.time()
    train_set = DataProcessor(root_path, train_data, train_labels)
    test_set = DataProcessor(root_path, validation_data, validation_labels)
    end = time.time()
    print(end - start)
    print(train_set.list_of_points.shape)
    print(test_set.list_of_points.shape)


