import os
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from utils import *
from utils.imresize import *
import cv2
import math

## RAFT
from glob import glob
from PIL import Image
from tqdm import tqdm

from ex_raft.raft.core.raft import RAFT
from ex_raft.raft.core.utils import flow_viz
from ex_raft.raft.core.utils.utils import InputPadder
from ex_raft.raft.config import RAFTConfig

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

## RAFT

# 학습 데이터 로더 클래스 정의
class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.scale_factor = args.scale_factor
        if args.task == 'SR':
            self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_train + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/'
            pass

        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    # 주어진 인덱스에 해당하는 데이터를 반환하는 메소드
    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        patchsize_hr = self.scale_factor * 32
        
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # 저해상도 이미지
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # 고해상도 이미지
            Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)

            Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    # 데이터셋의 총 아이템 수를 반환하는 메소드
    def __len__(self):
        return self.item_num

# 여러 테스트 세트 데이터 로더를 생성하는 함수
def MultiTestSetDataLoader(args):
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests

# 테스트 세트 데이터 로더 클래스 정의
class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name='ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    # 주어진 인덱스에 해당하는 데이터를 반환하는 메소드
    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr = np.transpose(Sr_SAI_cbcr, (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    # 데이터셋의 총 아이템 수를 반환하는 메소드
    def __len__(self):
        return self.item_num

# SAI 이미지를 좌우 대칭하는 함수
def flip_SAI(data, angRes):
    if len(data.shape) == 2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data

# 데이터와 레이블을 증강하는 함수
def augmentation(data, label):
    if random.random() < 0.5:  # W-V 방향으로 좌우 반전
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # W-V 방향으로 좌우 반전
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # U-V와 H-W를 전치
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label

# 이미지를 패치로 분할하는 함수
def extract_patches(data, patch_size):
    patches = []
    for i in range(0, data.shape[0], patch_size):
        for j in range(0, data.shape[1], patch_size):
            patch = data[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)

# 패치를 사용하여 이미지를 재구성하는 함수
def reconstruct_image(patches, angRes, patch_size):
    reconstructed_image = np.zeros((angRes * patch_size, angRes * patch_size))
    n_patches_per_row = angRes
    for idx, patch in enumerate(patches):
        i = (idx // n_patches_per_row) * patch_size
        j = (idx % n_patches_per_row) * patch_size
        reconstructed_image[i:i+patch_size, j:j+patch_size] = patch
    return reconstructed_image

# MIB 방식으로 이미지를 자르고 결합하는 함수
def cutmib(im1, im2, im1_mix, im2_mix, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = np.int(h_lr*cut_ratio), np.int(w_lr*cut_ratio)
    ch_hr, cw_hr = ch_lr*scale, cw_lr*scale
    cy_lr = np.random.randint(0, h_lr-ch_lr+1)
    cx_lr = np.random.randint(0, w_lr-cw_lr+1)
    cy_hr, cx_hr = cy_lr*scale, cx_lr*scale

    if np.random.random() < prob:
        if np.random.random() > 0.5:
            for i in range(an):
                im2[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = imresize(im1_mix[..., cy_hr:cy_hr+ch_hr, cx_hr:cx_hr+cw_hr], scalar_scale=1/scale)
        else:
            im2_aug = im2
            for i in range(an):
                im2_aug[i] = imresize(im1[i], scalar_scale=1/scale)
                im2_aug[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = im2_mix[..., cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr]
                im2 = im2_aug
        return im1, im2
    else: 
        return im1, im2

# Optical Flow를 사용하여 MIB 방식으로 이미지를 자르고 결합하는 함수
def op_cutmib(im1, im2, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = np.int(h_lr*cut_ratio), np.int(w_lr*cut_ratio)
    ch_hr, cw_hr = ch_lr*scale, cw_lr*scale
    cy_lr = np.random.randint(0, h_lr-ch_lr+1)
    cx_lr = np.random.randint(0, w_lr-cw_lr+1)
    cy_hr, cx_hr = cy_lr*scale, cx_lr*scale
    if np.random.random() < prob:
        if np.random.random() > 0.5:
            
            mix=[]
            for j in range(an):
                tmp = np.zeros((ch_hr,cw_hr),dtype=np.float32)
                
                deepF=cv2.optflow.createOptFlow_DeepFlow()
                flow=deepF.calc(im1[int((an+1)/2)], im1[j], None)
                
                for y in range(ch_hr):
                    for x in range(cw_hr):
                        dx,dy = flow[cy_hr+y, cx_hr+x].astype(np.float32)
                        if np.isnan(dx) or np.isnan(dy):
                            dx =0; dy = 0
                        
                        dx_=math.floor(dx)
                        dy_=math.floor(dy)
                        
                        if (cy_hr+y+dy+1>=im1[j].shape[0]) or (cx_hr+x+dx+1>=im1[j].shape[1]) or cy_hr+y+dy<0 or cx_hr+x+dx<0:                               
                            tmp[y,x]=im1[j,cy_hr+y,cx_hr+x] 
                        else: 
                            a=dx-dx_
                            b=dy-dy_
                            
                            q11=im1[j,cy_hr+y+dy_,cx_hr+x+dx_] 
                            q12=im1[j,cy_hr+y+dy_+1,cx_hr+x+dx_]
                            q21=im1[j,cy_hr+y+dy_,cx_hr+x+dx_+1]
                            q22=im1[j,cy_hr+y+dy_+1,cx_hr+x+dx_+1]
                            
                            tmp[y,x]=(q11 * (1 - a) * (1 - b) + q21 * a * (1 - b) + q12 * (1 - a) * b + q22 * a * b)     
            mix.append(tmp)        
            mix_=np.array(mix)
            mix_=mix_.mean(axis=0) 
            
            for i in range(an):  
                im2[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = imresize(mix_, scalar_scale=1/scale)
            
        else:
            im2_aug = im2
            
            mix=[]
            for j in range(an):
                j = int((an+1)/2)
                tmp = np.zeros((ch_lr,cw_lr),dtype=np.float32)
                
                deepF=cv2.optflow.createOptFlow_DeepFlow()
                flow=deepF.calc(im2[int((an+1)/2)], im2[j],None)
                
                for y in range(ch_lr):
                    for x in range(cw_lr):
                        dx,dy = flow[cy_lr+y, cx_lr+x].astype(np.float32)
                        if np.isnan(dx) or np.isnan(dy):
                            dx =0; dy = 0
                        dx_=math.floor(dx)
                        dy_=math.floor(dy)
                        
                        if (cy_lr+y+dy+1>=im2[j].shape[0]) or (cx_lr+x+dx+1>=im2[j].shape[1]) or cy_lr+y+dy<0 or cx_lr+x+dx<0:
                            tmp[y,x]=im2[j,cy_lr+y,cx_lr+x] 
                        else: 
                            a=dx-dx_
                            b=dy-dy_
                            
                            q11=im1[j,cy_lr+y+dy_,cx_lr+x+dx_]
                            q12=im1[j,cy_lr+y+dy_+1,cx_lr+x+dx_]
                            q21=im1[j,cy_lr+y+dy_,cx_lr+x+dx_+1]
                            q22=im1[j,cy_lr+y+dy_+1,cx_lr+x+dx_+1]
                            
                            tmp[y,x]=(q11 * (1 - a) * (1 - b) + q21 * a * (1 - b) + q12 * (1 - a) * b + q22 * a * b)                               
                                
            mix.append(tmp)        
            mix_=np.array(mix)
            mix_=mix_.mean(axis=0)
            
            for i in range(an):  
                im2_aug[i] = imresize(im1[i], scalar_scale=1/scale)
                im2_aug[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = mix_  
            
                im2 = im2_aug

        return im1, im2
    else: 
        return im1, im2

# 이미지를 디바이스에 로드하는 함수
def load_image(image, device):
    img = np.array(image).astype(np.uint8)
    
    if len(img.shape) == 2:  # 그레이스케일 이미지의 경우
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

# RAFT 설정
config = RAFTConfig(
    dropout=0,
    alternate_corr=False,
    small=False,
    mixed_precision=False
)

# RAFT를 사용하여 Optical Flow를 계산하는 함수
def raft(img1, img2):
    model = RAFT(config)
    model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = './ex_raft/raft-sintel.pth'
    
    ckpt = torch.load(weights_path, map_location=device)
    model.to(device)
    model.load_state_dict(ckpt)
    model.eval()
    
    image1 = load_image(img1, device)
    image2 = load_image(img2, device)
    
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    
    return flow_up[0].permute(1,2,0).cpu().detach().numpy()   

# RAFT를 사용하여 MIB 방식으로 이미지를 자르고 결합하는 함수
def raft_cutmib(im1, im2, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = np.int(h_lr*cut_ratio), np.int(w_lr*cut_ratio)
    ch_hr, cw_hr = ch_lr*scale, cw_lr*scale
    cy_lr = np.random.randint(0, h_lr-ch_lr+1)
    cx_lr = np.random.randint(0, w_lr-cw_lr+1)
    cy_hr, cx_hr = cy_lr*scale, cx_lr*scale
    if np.random.random() < prob:
        if np.random.random() > 0.5:
            
            mix=[]
            for j in range(an):
                tmp = np.zeros((ch_hr,cw_hr),dtype=np.float32)
                
                flow = raft(im1[int((an+1)/2)], im1[j])
                
                for y in range(ch_hr):
                    for x in range(cw_hr):
                        dx,dy = flow[cy_hr+y, cx_hr+x].astype(np.float32)
                        if np.isnan(dx) or np.isnan(dy):
                            dx =0; dy = 0
                        
                        dx_=math.floor(dx)
                        dy_=math.floor(dy)
                        
                        if (cy_hr+y+dy+1>=im1[j].shape[0]) or (cx_hr+x+dx+1>=im1[j].shape[1]) or cy_hr+y+dy<0 or cx_hr+x+dx<0:                               
                            tmp[y,x]=im1[j,cy_hr+y,cx_hr+x] 
                        else: 
                            a=dx-dx_
                            b=dy-dy_
                            
                            q11=im1[j,cy_hr+y+dy_,cx_hr+x+dx_] 
                            q12=im1[j,cy_hr+y+dy_+1,cx_hr+x+dx_]
                            q21=im1[j,cy_hr+y+dy_,cx_hr+x+dx_+1]
                            q22=im1[j,cy_hr+y+dy_+1,cx_hr+x+dx_+1]
                            
                            tmp[y,x]=(q11 * (1 - a) * (1 - b) + q21 * a * (1 - b) + q12 * (1 - a) * b + q22 * a * b)     
            mix.append(tmp)        
            mix_=np.array(mix)
            mix_=mix_.mean(axis=0) 
            
            for i in range(an):  
                im2[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = imresize(mix_, scalar_scale=1/scale)
            
        else:
            im2_aug = im2
            
            mix=[]
            for j in range(an):
                j = int((an+1)/2)
                tmp = np.zeros((ch_lr,cw_lr),dtype=np.float32)
                
                flow = raft(im2[int((an+1)/2)], im2[j])
                
                for y in range(ch_lr):
                    for x in range(cw_lr):
                        dx,dy = flow[cy_lr+y, cx_lr+x].astype(np.float32)
                        if np.isnan(dx) or np.isnan(dy):
                            dx =0; dy = 0
                        dx_=math.floor(dx)
                        dy_=math.floor(dy)
                        
                        if (cy_lr+y+dy+1>=im2[j].shape[0]) or (cx_lr+x+dx+1>=im2[j].shape[1]) or cy_lr+y+dy<0 or cx_lr+x+dx<0:
                            tmp[y,x]=im2[j,cy_lr+y,cx_lr+x] 
                        else: 
                            a=dx-dx_
                            b=dy-dy_
                            
                            q11=im1[j,cy_lr+y+dy_,cx_lr+x+dx_]
                            q12=im1[j,cy_lr+y+dy_+1,cx_lr+x+dx_]
                            q21=im1[j,cy_lr+y+dy_,cx_lr+x+dx_+1]
                            q22=im1[j,cy_lr+y+dy_+1,cx_lr+x+dx_+1]
                            
                            tmp[y,x]=(q11 * (1 - a) * (1 - b) + q21 * a * (1 - b) + q12 * (1 - a) * b + q22 * a * b)                               
                                
            mix.append(tmp)        
            mix_=np.array(mix)
            mix_=mix_.mean(axis=0)
            
            for i in range(an):  
                im2_aug[i] = imresize(im1[i], scalar_scale=1/scale)
                im2_aug[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = mix_  
            
                im2 = im2_aug

        return im1, im2
    else: 
        return im1, im2
