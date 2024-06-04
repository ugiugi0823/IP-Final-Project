'''
@inproceedings{LF_InterNet,
  title={Spatial-angular interaction for light field image super-resolution},
  author={Wang, Yingqian and Wang, Longguang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
  booktitle={European Conference on Computer Vision},
  pages={290--308},
  year={2020},
  organization={Springer}
}
'''
import torch
import torch.nn as nn


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in  # 각도 해상도 설정
        channels = 64  # 채널 수 설정
        self.factor = args.scale_factor  # 업스케일 팩터 설정
        n_groups, n_blocks = 4, 4  # 그룹 및 블록 수 설정

        # 특징 추출 모듈
        self.AngFE = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=int(self.angRes), stride=int(self.angRes), padding=0, bias=False))
        self.SpaFE = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=int(self.angRes), padding=int(self.angRes), bias=False))
        
        # 공간-각도 상호작용 모듈
        self.CascadeInterBlock = CascadeInterBlock(self.angRes, n_groups, n_blocks, channels)
        
        # 융합 및 재구성 모듈
        self.BottleNeck = BottleNeck(self.angRes, n_blocks, channels)
        self.ReconBlock = ReconBlock(self.angRes, channels, self.factor)

    def forward(self, x, Lr_info=None):
        x = SAI2MacPI(x, self.angRes)  # SAI 배열을 MacPI 형태로 변환
        xa = self.AngFE(x)  # 각도 특징 추출
        xs = self.SpaFE(x)  # 공간 특징 추출
        buffer_a, buffer_s = self.CascadeInterBlock(xa, xs)  # 상호작용 블록을 통해 공간-각도 특징 융합
        buffer_out = self.BottleNeck(buffer_a, buffer_s) + xs  # 융합된 특징과 초기 공간 특징 결합
        out = self.ReconBlock(buffer_out)  # 재구성 블록을 통해 HR 이미지 생성
        return out


class make_chains(nn.Module):
    def __init__(self, angRes, channels):
        super(make_chains, self).__init__()

        # 공간-각도 및 각도-공간 변환 모듈
        self.Spa2Ang = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes*angRes*channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        
        # 특징 결합 모듈
        self.AngConvSq = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.SpaConvSq = nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                            padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        buffer_ang1 = xa
        buffer_ang2 = self.ReLU(self.Spa2Ang(xs))  # 공간 특징을 각도 특징으로 변환
        buffer_spa1 = xs
        buffer_spa2 = self.Ang2Spa(xa)  # 각도 특징을 공간 특징으로 변환
        buffer_a = torch.cat((buffer_ang1, buffer_ang2), 1)  # 결합된 각도 특징
        buffer_s = torch.cat((buffer_spa1, buffer_spa2), 1)  # 결합된 공간 특징
        out_a = self.ReLU(self.AngConvSq(buffer_a)) + xa  # 각도 특징 결합 및 잔차 학습
        out_s = self.ReLU(self.SpaConvSq(buffer_s)) + xs  # 공간 특징 결합 및 잔차 학습
        return out_a, out_s


class InterBlock(nn.Module):
    def __init__(self, angRes, n_layers, channels):
        super(InterBlock, self).__init__()
        modules = []
        self.n_layers = n_layers
        for i in range(n_layers):
            modules.append(make_chains(angRes, channels))  # 다층 상호작용 체인 구성
        self.chained_layers = nn.Sequential(*modules)

    def forward(self, xa, xs):
        buffer_a = xa
        buffer_s = xs
        for i in range(self.n_layers):
            buffer_a, buffer_s = self.chained_layers[i](buffer_a, buffer_s)  # 각 층의 상호작용 수행
        out_a = buffer_a
        out_s = buffer_s
        return out_a, out_s


class CascadeInterBlock(nn.Module):
    def __init__(self, angRes, n_blocks, n_layers, channels):
        super(CascadeInterBlock, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(InterBlock(angRes, n_layers, channels))  # 다층 상호작용 블록 구성
        self.body = nn.Sequential(*body)

    def forward(self, buffer_a, buffer_s):
        out_a = []
        out_s = []
        for i in range(self.n_blocks):
            buffer_a, buffer_s = self.body[i](buffer_a, buffer_s)  # 각 블록의 상호작용 수행
            out_a.append(buffer_a)
            out_s.append(buffer_s)
        return torch.cat(out_a, 1), torch.cat(out_s, 1)  # 모든 블록의 출력을 결합


class BottleNeck(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck, self).__init__()

        # 병목 모듈
        self.AngBottle = nn.Conv2d(n_blocks*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.SpaBottle = nn.Conv2d((n_blocks+1)*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                    padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))  # 각도 병목 처리
        xs = torch.cat((xs, self.Ang2Spa(xa)), 1)  # 각도 병목 결과와 공간 특징 결합
        out = self.ReLU(self.SpaBottle(xs))  # 최종 융합 및 활성화
        return out


class ReconBlock(nn.Module):
    def __init__(self, angRes, channels, upscale_factor):
        super(ReconBlock, self).__init__()
        self.PreConv = nn.Conv2d(channels, channels * upscale_factor ** 2, kernel_size=3, stride=1,
                                 dilation=int(angRes), padding=int(angRes), bias=False)
        self.PixelShuffle = nn.PixelShuffle(upscale_factor)  # 업스케일링
        self.FinalConv = nn.Conv2d(int(channels), 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.angRes = angRes

    def forward(self, x):
        buffer = self.PreConv(x)  # 전처리 컨볼루션
        bufferSAI_LR = MacPI2SAI(buffer, self.angRes)  # MacPI를 SAI로 변환
        bufferSAI_HR = self.PixelShuffle(bufferSAI_LR)  # 픽셀 셔플링을 통해 업스케일링
        out = self.FinalConv(bufferSAI_HR)  # 최종 HR 이미지 생성
        return out


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])  # MacPI에서 SAI로 변환
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])  # SAI에서 MacPI로 변환
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self,args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()  # L1 손실 함수 사용

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)  # 손실 계산
        return loss
