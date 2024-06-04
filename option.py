import argparse

# ArgumentParser 객체 생성
parser = argparse.ArgumentParser()

# 작업(task) 인자 추가 (SR, RE 중 선택)
parser.add_argument('--task', type=str, default='SR', help='SR, RE')

# LF_SR 관련 인자 추가
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")  # 각도 해상도
parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")  # 스케일 팩터

# 모델 관련 인자 추가
parser.add_argument('--model_name', type=str, default='LFT', help="model name")  # 모델 이름
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")  # 사전 학습된 모델 체크포인트 사용 여부
parser.add_argument("--path_pre_pth", type=str, default='./pth/', help="path for pre model ckpt")  # 사전 학습된 모델 체크포인트 경로
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')  # 데이터셋 이름
parser.add_argument('--path_for_train', type=str, default='../../sojin_LFSR/data_for_training/')  # 학습 데이터 경로
parser.add_argument('--path_for_test', type=str, default='../../sojin_LFSR/data_for_test/')  # 테스트 데이터 경로
parser.add_argument('--path_log', type=str, default='./log/')  # 로그 경로

# 학습 설정 인자 추가
parser.add_argument('--batch_size', type=int, default=4)  # 배치 크기
parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')  # 초기 학습률
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')  # 가중치 감소율
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')  # 학습률 갱신 에포크 수
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')  # 감마 값
parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 50]')  # 총 학습 에포크 수

# 장치 및 기타 설정 인자 추가
parser.add_argument('--device', type=str, default='cuda:0')  # 사용 장치 설정
parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')  # 데이터 로더의 워커 수
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0)  # 로컬 랭크 설정

# 인자 파싱
args = parser.parse_args()

# 'SR' 작업일 경우 추가 설정
if args.task == 'SR':
    args.angRes_in = args.angRes  # 입력 각도 해상도 설정
    args.angRes_out = args.angRes  # 출력 각도 해상도 설정
    args.patch_size_for_test = 32  # 테스트 시 패치 크기
    args.stride_for_test = 16  # 테스트 시 스트라이드 크기
    args.minibatch_for_test = 1  # 테스트 시 미니배치 크기

# 사용하지 않는 인자 삭제
del args.angRes
