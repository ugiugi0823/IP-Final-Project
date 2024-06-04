'''
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE TPAMI},
    year      = {2022},
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 모델 클래스 정의
class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64  # 채널 수 설정
        n_group = 4  # 그룹 수 설정
        n_block = 4  # 블록 수 설정
        self.angRes = args.angRes_in  # 입력 각도 해상도 설정
        self.factor = args.scale_factor  # 스케일 팩터 설정
        # 초기 컨볼루션 레이어
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=self.angRes, padding=self.angRes, bias=False)
        # 분리 메커니즘 그룹 정의
        self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)
        # 업샘플링 레이어 정의
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(self.factor),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, info=None):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)  # 바이리니어 업샘플링
        x = SAI2MacPI(x, self.angRes)  # SAI 형식을 MacPI 형식으로 변환
        buffer = self.init_conv(x)  # 초기 컨볼루션 연산
        buffer = self.disentg(buffer)  # 분리 메커니즘 적용
        buffer_SAI = MacPI2SAI(buffer, self.angRes)  # MacPI 형식을 SAI 형식으로 변환
        out = self.upsample(buffer_SAI) + x_upscale  # 업샘플링 결과와 바이리니어 업샘플링 결과 더하기
        return out

# 분리 메커니즘을 여러 그룹으로 구성한 클래스 정의
class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group  # 그룹 수 설정
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)  # 그룹을 순차적으로 연결
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)  # 컨볼루션 레이어 정의

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)  # 각 그룹을 순차적으로 적용
        return self.conv(buffer) + x  # 잔차 학습 적용

# 분리 메커니즘의 단일 그룹 정의
class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block  # 블록 수 설정
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)  # 블록을 순차적으로 연결
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)  # 컨볼루션 레이어 정의

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)  # 각 블록을 순차적으로 적용
        return self.conv(buffer) + x  # 잔차 학습 적용

# 분리 메커니즘의 기본 블록 정의
class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels//4, channels//2  # 각 특징 추출 채널 수 설정

        # 공간적 특징 추출 레이어 정의
        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # 각도적 특징 추출 레이어 정의
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        # EPI(에피폴라 평면 이미지) 특징 추출 레이어 정의
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        # 추출된 특징을 결합하는 레이어 정의
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)  # 공간적 특징 추출
        feaAng = self.AngConv(x)  # 각도적 특징 추출
        feaEpiH = self.EPIConv(x)  # EPI 특징 추출 (수평)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)  # EPI 특징 추출 (수직)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)  # 추출된 모든 특징 결합
        buffer = self.fuse(buffer)  # 결합된 특징을 통해 최종 출력 생성
        return buffer + x  # 잔차 학습 적용

# 1D 픽셀 셔플링 레이어 정의
class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y

# MacPI 형식을 SAI 형식으로 변환하는 함수
def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out

# SAI 형식을 MacPI 형식으로 변환하는 함수
def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out

# 손실 함수 정의
class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()  # L1 손실 함수 사용

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)  # 예측 값과 실제 값 간의 L1 손실 계산
        return loss

# 가중치 초기화 함수
def weights_init(m):
    pass
