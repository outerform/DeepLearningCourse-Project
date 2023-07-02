# %%
import os
from posixpath import abspath
import torch
# import wandb

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt

# %%
from sklearn.model_selection import GroupKFold


# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=512, help='image size')
args = parser.parse_args()


# %%
IMG_SIZE = args.image_size
TRAIN_PATH = f'./tmp_{IMG_SIZE}/train/'
BATCH_SIZE = 16
EPOCHS = 10
n_fold = 5

folds = GroupKFold(n_splits=n_fold)
# %%
df = pd.read_csv('./siim-covid19-detection/train_image_level.csv')
df.head()


# %%
df.id = df.apply(lambda row: row.id.split('_')[0], axis=1)
df['path'] = df.apply(lambda row: TRAIN_PATH+row.id+'.png', axis=1)
df['image_level']  = df.apply(lambda row: row.label.split(' ')[0],axis=1)
df.head()

# %%
meta_df = pd.read_csv('./meta.csv')
# train_meta_df = meta_df.loc[meta_df.split=="train"]


# %%
train_meta_df = meta_df[meta_df.split=='train']

# %%
train_meta_df = train_meta_df.drop('split',axis=1)

# %%
train_meta_df.columns = ['id','dim0','dim1']

# %%
df = df.merge(train_meta_df, on='id',how='left')

# %%
def get_bbox(row):
    bboxes = []
    bbox = []
    labels  = row.label.split(' ')
    bboxes = []
    for i in range(0,len(labels), 6):
        # print(labels[i+2:i+6])
        bboxes.append(list(map(float,labels[i+2:i+6])))
    return np.array(bboxes)

        
        

# %%
def scale(row,bboxes):
    scalex = IMG_SIZE/row.dim1
    scaley = IMG_SIZE/row.dim0
    scaled_bboxes = np.zeros_like(bboxes)
    scaled_bboxes[:,[0,2]] = bboxes[:,[0,2]]*scalex
    scaled_bboxes[:,[1,3]] = bboxes[:,[1,3]]*scaley
    return scaled_bboxes

# %%
def yolo_format(bboxes):
    x = ((bboxes[:,0]+bboxes[:,2])/2)/IMG_SIZE
    y = ((bboxes[:,1]+bboxes[:,3])/2)/IMG_SIZE
    width = (bboxes[:,2]-bboxes[:,0])/IMG_SIZE
    height = (bboxes[:,3]-bboxes[:,1])/IMG_SIZE
    return np.column_stack([x,y,width,height]).astype(np.float32).reshape(-1,4)

# %%
def convert_to_yolo(row):
    bboxes = get_bbox(row)
    scaled_bboxes = scale(row,bboxes)
    yolo_bboxes = yolo_format(scaled_bboxes)
    # image_id = row.id
    # label = [0]*len(yolo_bboxes)
    return yolo_bboxes


# %%
for fold_id,(_,val_id) in enumerate(folds.split(df,groups=df.StudyInstanceUID.tolist())):
    df.loc[val_id,'fold_id'] = fold_id

for fold in range(n_fold):
    train_df = df.loc[df.fold_id != fold]
    valid_df = df.loc[df.fold_id == fold]

    # %%
    train_df.loc[:,'split'] = 'train'
    valid_df.loc[:,'split'] = 'valid'

    # %%
    df = pd.concat([train_df, valid_df]).reset_index(drop=True)

    # %%
    print(f'Size of dataset: {len(df)}, training images: {len(train_df)}. validation images: {len(valid_df)}')

    data_path = f'data_{IMG_SIZE}/fold{fold}'
    train_path = f'{data_path}/train'
    valid_path = f'{data_path}/valid'

    # # %%
    # os.makedirs(f'data_{IMG_SIZE}/fold{fold}/train/images', exist_ok=True)
    # os.makedirs(f'data_{IMG_SIZE}/fold{fold}/valid/images', exist_ok=True)
    os.makedirs(f'{train_path}/images', exist_ok=True)
    os.makedirs(f'{valid_path}/images', exist_ok=True)
    os.makedirs(f'{train_path}/labels', exist_ok=True)
    os.makedirs(f'{valid_path}/labels', exist_ok=True)
    

    # %%
    for i in tqdm(range(len(df))):
        row = df.loc[i]
        if row.split == 'train':
            copyfile(row.path, f'{train_path}/images/{row.id}.png')
        else:
            copyfile(row.path, f'{valid_path}/images/{row.id}.png')

    import yaml

    #%%
    data_yaml = dict(
        train = os.path.abspath(f'{train_path}/images'),
        val = os.path.abspath(f'{valid_path}/images'),
        nc = 2,
        names = ['none', 'opacity']
    )

    with open(f'{data_path}/data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)
        
    # %%
    for i in tqdm(range(len(df))):
        row = df.loc[i]
        # print(row.split)
        label = row.image_level
        if row.split == 'train':
            label_path = f'{train_path}/labels/{row.id}.txt'
        else:
            label_path = f'{valid_path}/labels/{row.id}.txt'
        
        if label == 'opacity':
            yoloboxes = convert_to_yolo(row)
            with open(label_path, 'w') as f:
                for bbox in yoloboxes:
                    f.write(f'1 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')




