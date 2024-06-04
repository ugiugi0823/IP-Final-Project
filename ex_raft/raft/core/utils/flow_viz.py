# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03


import numpy as np

# Optical Flow 시각화를 위한 컬러 휠을 생성하는 함수
def make_colorwheel():
    """
    Optical Flow 시각화를 위한 컬러 휠을 생성합니다. 해당 방법론은 아래 논문에서 제시되었습니다:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    코드는 Daniel Scharstein의 C++ 소스 코드와 Deqing Sun의 Matlab 소스 코드를 따릅니다.

    반환값:
        np.ndarray: 컬러 휠
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # 빨강-노랑
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col + RY
    # 노랑-초록
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    # 초록-시안
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col + GC
    # 시안-파랑
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    # 파랑-자주
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col + BM
    # 자주-빨강
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

# Flow 컴포넌트 u와 v에 컬러 휠을 적용하는 함수
def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    (가능하면 클리핑된) flow 컴포넌트 u와 v에 컬러 휠을 적용합니다.

    이 코드는 Daniel Scharstein의 C++ 소스 코드 및 Deqing Sun의 Matlab 소스 코드를 따릅니다.

    인자:
        u (np.ndarray): 입력 가로 방향 flow, 크기 [H,W]
        v (np.ndarray): 입력 세로 방향 flow, 크기 [H,W]
        convert_to_bgr (bool, 선택): 출력 이미지를 BGR로 변환할지 여부. 기본값은 False.

    반환값:
        np.ndarray: 크기 [H,W,3]의 Flow 시각화 이미지
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # 크기 [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # 범위를 벗어나는 경우
        # 2-i를 통해 BGR 대신 RGB로 변환
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image

# 두 개의 Flow 컴포넌트를 이용해 이미지를 생성하는 함수
def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    2차원 flow 이미지를 입력으로 받아들입니다.

    인자:
        flow_uv (np.ndarray): 크기 [H,W,2]의 Flow UV 이미지
        clip_flow (float, 선택): flow 값의 최대치를 클립합니다. 기본값은 None.
        convert_to_bgr (bool, 선택): 출력 이미지를 BGR로 변환할지 여부. 기본값은 False.

    반환값:
        np.ndarray: 크기 [H,W,3]의 Flow 시각화 이미지
    """
    assert flow_uv.ndim == 3, '입력 flow는 3차원이여야 합니다'
    assert flow_uv.shape[2] == 2, '입력 flow는 크기 [H,W,2]여야 합니다'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)
