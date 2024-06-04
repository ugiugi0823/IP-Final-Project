import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from option import args
from einops import rearrange
import xlwt
import torch.nn.functional as F

# 엑셀 파일을 생성하고 데이터를 기록하는 클래스
class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()  # 새로운 엑셀 파일 생성
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)  # 새로운 시트 추가
        # 시트의 첫 행에 컬럼 제목 설정
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        # 컬럼 너비 설정
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1  # 데이터 기록을 위한 행 번호 초기화

    # PSNR 및 SSIM 값을 시트에 기록하는 메소드
    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' PSNR 및 SSIM 값을 저장 '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i])

        # 각 테스트에 대한 평균 PSNR 및 SSIM 계산
        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    # 개별 테스트 결과를 시트에 기록하는 메소드
    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' PSNR 및 SSIM 값을 저장 '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.sum = self.sum + 1

# 로그 파일을 설정하는 함수
def get_logger(log_dir, args):
    '''로그 설정'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)  # 로그 레벨 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 로그 형식 설정
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# 디렉토리를 생성하는 함수
def create_dir(args):
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)  # 로그 디렉토리 생성
    if args.task == 'SR':
        task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x'
    
    log_dir = log_dir.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)  # 태스크 디렉토리 생성
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)  # 데이터 이름 디렉토리 생성
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)  # 모델 이름 디렉토리 생성

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)  # 체크포인트 디렉토리 생성

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)  # 결과 디렉토리 생성

    return log_dir, checkpoints_dir, results_dir

# 로그 클래스 정의
class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    # 로그 메시지를 출력하는 메소드
    def log_string(self, str):
        if args.local_rank <= 0:
            self.logger.info(str)
            print(str)

# PSNR 및 SSIM을 계산하는 함수
def cal_metrics(args, label, out):
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy())
                if args.task == 'RE':
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True,
                                                                  sigma=1.5, use_sample_covariance=False)
                else:
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True, data_range=1)

    if args.task == 'RE':
        for u in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
            for v in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
                PSNR[:, u, v] = 0
                SSIM[:, u, v] = 0

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean

# 이미지를 확장하는 함수
def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out

# LF 이미지를 패치로 나누는 함수
def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF

# 패치를 사용하여 LF 이미지를 재구성하는 함수
def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF

# RGB 이미지를 YCbCr 이미지로 변환하는 함수
def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y

# YCbCr 이미지를 RGB 이미지로 변환하는 함수
def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y
