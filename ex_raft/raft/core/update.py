import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)  # 첫 번째 2D 합성곱 레이어 정의
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)  # 두 번째 2D 합성곱 레이어 정의
        self.relu = nn.ReLU(inplace=True)  # ReLU 활성화 함수 정의

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))  # 순전파: conv1 -> ReLU -> conv2


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)  # 업데이트 게이트 합성곱 정의
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)  # 리셋 게이트 합성곱 정의
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)  # 새로운 상태 계산을 위한 합성곱 정의

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)  # 입력 및 이전 상태를 결합

        z = torch.sigmoid(self.convz(hx))  # 업데이트 게이트 계산
        r = torch.sigmoid(self.convr(hx))  # 리셋 게이트 계산
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))  # 새로운 상태 계산

        h = (1-z) * h + z * q  # 최종 상태 계산
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))  # 첫 번째 업데이트 게이트(가로 방향)
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))  # 첫 번째 리셋 게이트(가로 방향)
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))  # 첫 번째 새로운 상태(가로 방향)

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))  # 두 번째 업데이트 게이트(세로 방향)
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))  # 두 번째 리셋 게이트(세로 방향)
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))  # 두 번째 새로운 상태(세로 방향)

    def forward(self, h, x):
        # 가로 방향 계산
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # 세로 방향 계산
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)  # 상관관계 특징을 위한 합성곱 레이어
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)  # 흐름 특징을 위한 첫 번째 합성곱 레이어
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)  # 흐름 특징을 위한 두 번째 합성곱 레이어
        self.conv = nn.Conv2d(128, 80, 3, padding=1)  # 결합된 특징을 위한 합성곱 레이어

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))  # 상관관계 특징 추출
        flo = F.relu(self.convf1(flow))  # 흐름 특징 추출
        flo = F.relu(self.convf2(flo))  # 추가 흐름 특징 추출
        cor_flo = torch.cat([cor, flo], dim=1)  # 특징 결합
        out = F.relu(self.conv(cor_flo))  # 최종 특징 추출
        return torch.cat([out, flow], dim=1)  # 결합된 결과 반환


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)  # 상관관계 특징을 위한 첫 번째 합성곱 레이어
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)  # 상관관계 특징을 위한 두 번째 합성곱 레이어
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)  # 흐름 특징을 위한 첫 번째 합성곱 레이어
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)  # 흐름 특징을 위한 두 번째 합성곱 레이어
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)  # 결합된 특징을 위한 합성곱 레이어

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))  # 상관관계 특징 추출
        cor = F.relu(self.convc2(cor))  # 추가 상관관계 특징 추출
        flo = F.relu(self.convf1(flow))  # 흐름 특징 추출
        flo = F.relu(self.convf2(flo))  # 추가 흐름 특징 추출

        cor_flo = torch.cat([cor, flo], dim=1)  # 특징 결합
        out = F.relu(self.conv(cor_flo))  # 최종 특징 추출
        return torch.cat([out, flow], dim=1)  # 결합된 결과 반환


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)  # SmallMotionEncoder 인스턴스 생성
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)  # ConvGRU 인스턴스 생성
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)  # FlowHead 인스턴스 생성

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)  # 움직임 특징 추출
        inp = torch.cat([inp, motion_features], dim=1)  # 입력과 움직임 특징 결합
        net = self.gru(net, inp)  # GRU 네트워크 업데이트
        delta_flow = self.flow_head(net)  # 흐름 변화 추출

        return net, None, delta_flow  # 결과 반환


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)  # BasicMotionEncoder 인스턴스 생성
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)  # SepConvGRU 인스턴스 생성
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)  # FlowHead 인스턴스 생성

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))  # 마스크 생성기를 위한 합성곱 레이어 정의

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)  # 움직임 특징 추출
        inp = torch.cat([inp, motion_features], dim=1)  # 입력과 움직임 특징 결합

        net = self.gru(net, inp)  # GRU 네트워크 업데이트
        delta_flow = self.flow_head(net)  # 흐름 변화 추출

        # 그라디언트 균형을 맞추기 위해 마스크 스케일링
        mask = .25 * self.mask(net)
        return net, mask, delta_flow  # 결과 반환
