# %%
device = 'cuda:0'

# %%
import random
import numpy as np
import torch
import os

# %%
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed()


# %%
# hyperparameters
from scipy import optimize


class args:
    batch_size = 12
    n_worker = 8

    image_size = 512
    arch_name = 'tf_efficientnetv2_l.in21k'
    epochs = 20
    lr = 1e-4
    drop_rate = 0.45
    drop_path_rate = 0.1

    loss_fn = "FocalLoss"
    focal_alpha = 0.6
    focal_gamma = 1.8
    aux_loss = 'binary_cross_entropy'


    loss1_coef = 1
    optimizer = 'AdamW'
    scheduler = 'CosineAnnealingLR'
    scheduler_warmup = None # "GradualWarmupSchedulerV3"
    warmup_factor = 2
    warmup_epo = 5
    T_max = epochs - 1
    weight_decay = 2e-3

    save_path = './models_2/'
    earlystop_patience = 5



# %%
import pandas as pd
train_label_image = pd.read_csv('./siim-covid19-detection/train_image_level.csv')
train_label_study = pd.read_csv('./siim-covid19-detection/train_study_level.csv')

# train_label

# %%
study_id = set()
boxes_study_id= set()
uniqued_train_label_image = pd.DataFrame(columns = train_label_image.columns)

for idx,image_id,boxes,label,StudyInstanceUID in train_label_image.itertuples():
    # break
    if StudyInstanceUID not in study_id:
        study_id.add(StudyInstanceUID)
        # check if opacity in label
        if label.find('none') == -1:
            boxes_study_id.add(StudyInstanceUID)
        # print(uniqued_train_label_image.columns)
        # print(train_label_image.iloc[idx])
        uniqued_train_label_image = uniqued_train_label_image.append(train_label_image.iloc[idx])
        # print(uniqued_train_label_image.columns)
        # break
    else:
        if StudyInstanceUID in boxes_study_id:
            continue
        elif label.find('none') == -1:
            boxes_study_id.add(StudyInstanceUID)
            uniqued_train_label_image.loc[uniqued_train_label_image['StudyInstanceUID'] == StudyInstanceUID] = [image_id,boxes,label,StudyInstanceUID]

uniqued_train_label_image = uniqued_train_label_image.reset_index(drop=True)

    
        

# %%
train_label_study['StudyInstanceUID'] = train_label_study['id'].apply(lambda x: x.replace('_study',''))

# %%
train_df = pd.merge(uniqued_train_label_image,train_label_study,on='StudyInstanceUID')

# %%
train_df.rename(columns={'id_x':'id_image'},inplace=True)
train_df.drop(['id_y'],axis=1,inplace=True)


# %%
train_df['id_image'] = train_df['id_image'].apply(lambda x: x.replace('_image',''))

# %%
train_df

# %%
train_df['filepath'] = './tmp/train/'+train_df['id_image']+'.png'

# %%
from sklearn.model_selection import GroupKFold
folds = GroupKFold(n_splits=5)

# %%
# add column for fold
train_df['fold_id'] = -1
for fold_id,(train_idx,valid_idx) in enumerate(folds.split(train_df,groups=train_df.StudyInstanceUID.tolist())):
    train_df.loc[valid_idx,'fold_id'] = fold_id

# %%
train_df

# %%
rotate_limit = 20
scale_limit = 0.7
shift_limit =  0.3
num_holes = 8
max_h_size = 0.05
max_w_size = 0.05

# %% [markdown]
# 

# %%
import albumentations
from albumentations.pytorch import ToTensorV2
from albumentations import RandomCrop,HorizontalFlip,VerticalFlip,Rotate,RandomBrightnessContrast,\
    RandomResizedCrop,Normalize,Resize,Compose,GaussianBlur,RandomBrightness,RandomContrast,RandomGamma,RandomRotate90,Transpose,\
    ShiftScaleRotate,Blur,OpticalDistortion,GridDistortion,HueSaturationValue,IAAAdditiveGaussianNoise,IAAPerspective,RandomSizedCrop,\
    RandomShadow,RandomSnow,RandomRain,RandomFog,CenterCrop,CoarseDropout,ChannelShuffle,ToGray,Cutout,PadIfNeeded,RandomCrop,VerticalFlip,HorizontalFlip,\
    Transpose,RandomRotate90,ShiftScaleRotate,ElasticTransform,GridDistortion,OpticalDistortion,RandomSizedCrop,HueSaturationValue,RGBShift,RandomBrightnessContrast,\
    RandomGamma,CLAHE,Blur,MedianBlur,MotionBlur,GaussNoise,GaussianBlur,RGBShift,RandomBrightnessContrast,IAAEmboss,IAASharpen,IAASuperpixels,RandomFog,RandomRain,\
    RandomShadow,RandomSnow,RandomSunFlare
transform_train = albumentations.Compose([
    RandomCrop(args.image_size,args.image_size),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(p=0.7,shift_limit=shift_limit,scale_limit=scale_limit,rotate_limit=rotate_limit),
    Cutout(p=0.7,num_holes=8,max_h_size=int(args.image_size*max_h_size),max_w_size=int(args.image_size*max_w_size))
    ],
    additional_targets={'mask':'image'})
transform_train_image = albumentations.Compose([
    RandomBrightnessContrast(p=0.1,brightness_limit=0.3,contrast_limit=0.3),
    HueSaturationValue(p=0.1,hue_shift_limit=20,sat_shift_limit=50,val_shift_limit=40),
    Normalize()])
transform_valid = albumentations.Compose([
    Resize(args.image_size,args.image_size),
    Normalize()],
)



# %%
import cv2
from PIL import Image
from torchvision.transforms import ToTensor,Normalize,Compose,Resize
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,df,transform=None,mode:str='train'):
        self.df = df
        self.transform = transform
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert('RGB')
        if self.mode == 'train':
            mask_img = Image.open(row.filepath.replace('train','train_mask')).convert('RGB')
            transformed = self.transform[0](image=np.array(img),mask=np.array(mask_img))
            mask_img = transformed['mask'].transpose(2,0,1)/255
            img = self.transform[1](image=transformed['image'])['image'].transpose(2,0,1)
            # print(row)
            label = torch.tensor(row[['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']]).float()
            return img,mask_img,label
        elif self.mode == 'valid':
            img = self.transform(image=np.array(img))['image'].transpose(2,0,1)
            label = torch.tensor(row[['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']]).float()
            return img,label
        elif self.mode == 'test':
            img = self.transform(image=np.array(img))['image'].transpose(2,0,1)
            return img
        else:
            raise ValueError('mode must be train or valid or test')
        

# %%
train_dataset = MyDataset(train_df,transform=[transform_train,transform_train_image],mode='train')

# %% [markdown]
# Model

# %%
from torch import adaptive_max_pool1d
from torch.utils.data import DataLoader
#effnetv2
import torch.nn as nn
import timm
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import AvgPool2d,AdaptiveAvgPool2d

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
    
    def forward(self,inputs,targets):
        if self.logits:
            loss = nn.BCEWithLogitsLoss()(inputs,targets)
        else:
            loss = nn.CrossEntropyLoss()(inputs,targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss
        if self.reduce is not None:
            return torch.__getattribute__(self.reduce)(F_loss)
        else:
            return F_loss

class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
        
class Effnetv2(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.model = timm.create_model(args.arch_name,pretrained=pretrained,num_classes=4,drop_rate = args.drop_rate,drop_path_rate=args.drop_path_rate)
        self.logit = nn.Linear(self.model.classifier.in_features,4)
        self.preprocess = nn.Sequential(
            self.model.conv_stem,
            self.model.bn1,
            Swish()
        )
        self.blocks = self.model.blocks
        self.mask = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )
        self.classifier = nn.Sequential(
             self.model.conv_head,
                self.model.bn2,
                Swish(),
        )



    def forward(self,x):
        batch_size = len(x)
        x = self.preprocess(x)
        for block in self.blocks[:5]:
            x = block(x)
        mask = self.mask(x)
        for block in self.blocks[5:]:
            x = block(x)
        x = self.classifier(x)
        x = AdaptiveAvgPool2d((1,1))(x)
        return self.logit(x.view(batch_size,-1)),mask

# %%
import wandb
wandb.init(project='siim-covid19-detection',name='effnetv2')


# %%
from enum import auto
from numpy import average
from torch.optim import AdamW
from torch.cuda.amp import GradScaler,autocast
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score

scaler = GradScaler()
class Trainer:
    def __init__(self,model,train_dataloader, valid_dataloader) -> None:
        self.model = model

    def accuracy(outputs,targets,logits = True):
        outputs = torch.argmax(outputs,dim=1)
        if logits:
            targets = torch.argmax(targets,dim=1)
        return (outputs == targets).float().mean()
    
    def train(self,train_dataloader,optimizer, loss_fn = "BCEWithLogitsLoss",aux_loss = "binary_cross_entropy",_scheduler = 'CosineAnnealingLR',T_max = 10,eta_min = 1e-6,valid_dataloader = None,device = 'cuda:0',aux_weight = 0.1,epochs = 10,verbose = True,save_path = None,save = True,save_best = True,early_stopping = True,patience = 5,):
        
        if loss_fn == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_fn == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'FocalLoss':
            criterion = FocalLoss()
        else:
            raise ValueError('loss_fn must be BCEWithLogitsLoss or CrossEntropyLoss or FocalLoss')
        
        if aux_loss == 'binary_cross_entropy':
            aux_criterion = F.binary_cross_entropy_with_logits
        else:
            raise ValueError('aux_loss must be binary_cross_entropy')
        model=self.model
        model.train()
        losses = []
        aux_losses = []
        loss0s = []
        accs = []
        with tqdm(enumerate(train_dataloader),total=len(train_dataloader)) as pbar:
            for idx,(img,mask_img,label) in pbar:
                img = img.to(device)
                # print(img)
                mask_img = mask_img.to(device)
                mask_img = mask_img[:,0:1,:,:]
                # print(mask_img)
                mask_img = F.interpolate(mask_img,(32,32), mode='bilinear', align_corners=False)
                
                label = label.to(device)
                with autocast():
                    optimizer.zero_grad()
                    logits,mask = model(img)
                    if loss_fn == 'CrossEntropyLoss':
                        loss0 = criterion(logits,label.argmax(-1))
                    else:
                        loss0 = criterion(logits,label)
                    aux_loss = aux_criterion(mask,mask_img)*aux_weight
                    scaler.scale(loss0+aux_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    accs.append(Trainer.accuracy(logits,label).item())
                    aux_losses.append(aux_loss.item())
                    loss0s.append(loss0.item())
                    losses.append((loss0+aux_loss).item())
                    pbar.set_postfix(loss=losses[-1])
                    if (idx+1) % 100 == 0:
                        # pbar.write(f'idx:{idx+1},loss:{np.mean(losses)},acc:{np.mean(accs)}')
                        wandb.log({"train_loss": np.mean(losses),"train_acc":np.mean(accs)})
        return losses,accs,optimizer

    def valid(self,valid_dataloader,loss_fn = "BCEWithLogitsLoss",device = 'cuda:0'):
        model = self.model
        model.eval()
        losses = []
        accs = []
        preds = []
        targets = []
        if loss_fn == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_fn == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'FocalLoss':
            criterion = FocalLoss()
        else:
            raise ValueError('loss_fn must be BCEWithLogitsLoss or CrossEntropyLoss or FocalLoss')
        with torch.no_grad():
            with tqdm(enumerate(valid_dataloader),total=len(valid_dataloader)) as pbar:
                for idx,(img,label) in pbar:
                    img = img.to(device)
                    label = label.to(device)
                    with torch.no_grad():
                        logits,mask = model(img)
                        if loss_fn == 'CrossEntropyLoss':
                            loss = criterion(logits,label.argmax(-1))
                        else:
                            loss = criterion(logits,label)
                        # accuracy = lambda x,y: (x.argmax(-1) == y.argmax(-1)).float().mean()
                        accs.append(Trainer.accuracy(logits,label).item())
                        losses.append(loss.item())
                        preds.append(logits.cpu().numpy())
                        targets.append(label.cpu().numpy())
                        pbar.set_postfix(loss=loss.item(),acc=accs[-1])
        preds = np.concatenate(preds,axis=0)
        targets = np.concatenate(targets,axis=0)
        ap = average_precision_score(targets,preds)
        wandb.log({"valid_loss": np.mean(losses),"valid_acc":np.mean(accs),"average_precision_score":ap})
        return losses,accs,np.concatenate(preds,axis=0),ap

        
        

# %%

from math import e
from torch.optim import AdamW,Adam,SGD
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau,CosineAnnealingWarmRestarts
import shutil

import transformers

def get_dataloader(fold,df):
    train_df = df[df.fold_id != fold]
    valid_df = df[df.fold_id == fold]
    train_dataset = MyDataset(train_df,transform = [transform_train,transform_train_image],mode = 'train')
    valid_dataset = MyDataset(valid_df,transform = transform_valid,mode = 'valid')
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.n_worker,pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.n_worker,pin_memory=True)

    return train_dataloader,valid_dataloader


for fold in range(5):
    save_path_fold = os.path.join(args.save_path,f'fold_{fold}')
    os.makedirs(save_path_fold,exist_ok=True)
    model = Effnetv2()
    model.to(device)
    train_dataloader,valid_dataloader = get_dataloader(fold,train_df)
    trainer = Trainer(model,train_dataloader,valid_dataloader)
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(),lr=args.lr
                            ,weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(),lr=args.lr
                            ,weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(),lr=args.lr)
    else:
        raise ValueError('optimizer must be AdamW or Adam or SGD')

    # if args.scheduler == 'CosineAnnealingWarmRestarts':
    #     scheduler = CosineAnnealingWarmRestarts(optimizer)
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer,T_max=args.T_max)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=args.factor,patience=args.patience,verbose=True)
    else:
        raise ValueError('scheduler must be CosineAnnealingWarmRestarts or CosineAnnealingLR or ReduceLROnPlateau')
    best = 0
    best_epoch = 0
    early_stop = 0

    for epoch in range(args.epochs):
        train_losses,train_accs,optimizer = trainer.train(train_dataloader,valid_dataloader=valid_dataloader,device=device,loss_fn=args.loss_fn,aux_loss=args.aux_loss,optimizer=optimizer,aux_weight=args.loss1_coef)
        valid_losses,valid_accs,valid_preds,ap = trainer.valid(valid_dataloader,device=device,loss_fn=args.loss_fn)
        scheduler.step()
        if ap > best:
            early_stop = 0
            best = ap
            best_epoch = epoch
        else:
            early_stop += 1
        if early_stop >= args.earlystop_patience:
            break
        print(f'epoch:{epoch},train_loss:{np.mean(train_losses)},train_acc:{np.mean(train_accs)},valid_loss:{np.mean(valid_losses)},valid_acc:{np.mean(valid_accs)}')
        torch.save(model.state_dict(),os.path.join(save_path_fold,f'epoch{epoch}.pth'))
    shutil.copy(os.path.join(save_path_fold,f'epoch{best_epoch}.pth'),os.path.join(save_path_fold,f'best_epoch{best_epoch}_ap{best}.pth'))
    wandb.log({"best_epoch": best_epoch,"best_ap":best})

# %%
# valid_losses,valid_accs,valid_preds,ap = trainer.valid(valid_dataloader,device=device,loss_fn=args.loss_fn)

# %%
# for x in timm.list_models(pretrained=True):
#     if x.find('efficientnetv2') != -1:
#         print(x)

# %%
# from sklearn.metrics import average_precision_score
# x = np.array([[0.2,0.0],[0.3,0.7]])
# y = np.array([[0,1],[1,0]])
# average_precision_score(y,x)
#get number of files
# len(os.listdir('./data/train/labels'))

# %%



