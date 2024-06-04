import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate

class InputPadder:
    """ 이미지를 8로 나누어 떨어지도록 패딩하는 클래스 """
    def __init__(self, dims, mode='sintel'):
        # 이미지의 높이와 너비 설정
        self.ht, self.wd = dims[-2:]
        # 높이와 너비가 8로 나누어 떨어지도록 필요한 패딩 계산
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        # 모드에 따라 패딩 설정
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        # 주어진 입력들에 대해 패딩 적용
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        # 패딩을 제거하고 원래 크기로 복원
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    # 흐름 필드(flow)를 numpy 배열로 변환
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    # 흐름 필드의 높이와 너비를 가져옴
    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    # 새로운 좌표 계산
    x1 = x0 + dx
    y1 = y0 + dy
    
    # 1차원 배열로 변환
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    # 유효한 좌표 필터링
    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    # 최근접 이웃 보간법으로 흐름 필드를 다시 샘플링
    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    # 다시 흐름 필드로 병합
    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ grid_sample을 위한 래퍼, 픽셀 좌표를 사용 """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2 * xgrid / (W-1) - 1
    ygrid = 2 * ygrid / (H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    # 좌표 그리드 생성
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def upflow8(flow, mode='bilinear'):
    # 흐름 필드를 8배로 업샘플링
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
