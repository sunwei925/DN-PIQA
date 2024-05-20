import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from PIL import Image


class PIQA_pair(Dataset):
    def __init__(self, data_dir, feature_dir, face_dir, csv_path, transform, database):
        self.database = database

        column_names = ['IMAGE PATH','JOD','JOD STD','CI LOW','CI HIGH','CI RANGE','QUALITY LEVEL','IMAGE','SCENE','CONDITION']
        column_names_second = [i + ' second' for i in column_names]

        tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names+column_names_second, index_col=False, encoding="utf-8-sig")
        self.X_train = tmp_df[['IMAGE']]
        self.Y_train = tmp_df['JOD']
        self.Y_train_std = tmp_df['JOD STD']
        self.X_train_second = tmp_df[['IMAGE second']]
        self.Y_train_second = tmp_df['JOD second']
        self.Y_train_std_second = tmp_df['JOD STD second']

        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.face_dir = face_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):
        # the first image
        image_name = self.X_train.iloc[index,0]
        path1 = os.path.join(self.data_dir,image_name)
        path2 = os.path.join(self.face_dir,image_name.split('.')[0]+'.png')

        LIQE_feature_path = os.path.join(self.feature_dir, image_name.split('.')[0]+'.npy')

        img1 = Image.open(path1)
        img1 = img1.convert('RGB')
        img2 = Image.open(path2)
        img2 = img2.convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        y_mos = self.Y_train.iloc[index]

        y_label = torch.FloatTensor(np.array(float(y_mos)))


        LIQE_feature = np.load(LIQE_feature_path)
        LIQE_feature = torch.from_numpy(LIQE_feature)
        LIQE_feature = LIQE_feature.squeeze()

        # the second image
        image_name_second = self.X_train_second.iloc[index,0]
        path1_second = os.path.join(self.data_dir,image_name_second)
        path2_second = os.path.join(self.face_dir,image_name_second.split('.')[0]+'.png')

        LIQE_feature_path_second = os.path.join(self.feature_dir, image_name_second.split('.')[0]+'.npy')

        img1_second = Image.open(path1_second)
        img1_second = img1_second.convert('RGB')
        img2_second = Image.open(path2_second)
        img2_second = img2_second.convert('RGB')

        if self.transform is not None:
            img1_second = self.transform(img1_second)
            img2_second = self.transform(img2_second)

        y_mos_second = self.Y_train_second.iloc[index]

        y_label_second = torch.FloatTensor(np.array(float(y_mos_second)))


        LIQE_feature_second = np.load(LIQE_feature_path_second)
        LIQE_feature_second = torch.from_numpy(LIQE_feature_second)
        LIQE_feature_second = LIQE_feature_second.squeeze()


        return img1, img2, LIQE_feature, y_label, img1_second, img2_second, LIQE_feature_second, y_label_second


    def __len__(self):
        return self.length










class PIQA(Dataset):
    def __init__(self, data_dir, feature_dir, face_dir, csv_path, transform, database):
        self.database = database
        if self.database == 'PIQ_train':
            column_names = ['IMAGE PATH','JOD','JOD STD','CI LOW','CI HIGH','CI RANGE','QUALITY LEVEL','IMAGE','SCENE','CONDITION']
        elif self.database == 'PIQ_validation':
            column_names = ['IMAGE PATH', 'JOD', 'JOD STD', 'CI LOW', 'CI HIGH', 'CI RANGE', 'QUALITY LEVEL', 'CLUSTER', 'TOTAL COMPARISONS', 'IMAGE', 'SCENE', 'ATTRIBUTE', 'SCENE IDX', 'CONDITION', 'SPLIT']

        tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X_train = tmp_df[['IMAGE']]
        self.Y_train = tmp_df['JOD']
        self.scene = tmp_df['SCENE']

        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.face_dir = face_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):
        image_name = self.X_train.iloc[index,0]
        path1 = os.path.join(self.data_dir,image_name)
        path2 = os.path.join(self.face_dir,image_name.split('.')[0]+'.png')

        LIQE_feature_path = os.path.join(self.feature_dir, image_name.split('.')[0]+'.npy')

        img1 = Image.open(path1)
        img1 = img1.convert('RGB')
        img2 = Image.open(path2)
        img2 = img2.convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        y_mos = self.Y_train.iloc[index]

        y_label = torch.FloatTensor(np.array(float(y_mos)))
        
        scene = self.scene.iloc[index]


        LIQE_feature = np.load(LIQE_feature_path)
        LIQE_feature = torch.from_numpy(LIQE_feature)
        LIQE_feature = LIQE_feature.squeeze()


        return img1, img2, LIQE_feature, y_label, scene


    def __len__(self):
        return self.length

