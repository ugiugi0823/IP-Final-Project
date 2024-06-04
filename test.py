import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import MultiTestSetDataLoader
from collections import OrderedDict
from train import test

def main(args):
    ''' 결과 저장을 위한 디렉토리 생성 '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU 또는 Cuda 설정 '''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' 테스트 데이터 로딩 '''
    print('\n테스트 데이터셋 로드 중 ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("테스트 데이터의 수: %d" % length_of_tests)

    ''' 모델 로딩 '''
    print('\n모델 초기화 중 ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' 사전 학습된 모델 로드 '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # 'module.'을 추가
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            print('사전 학습된 모델을 사용합니다!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            net.load_state_dict(new_state_dict)
            print('사전 학습된 모델을 사용합니다!')
            pass
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' 매개변수 출력 '''
    print('매개변수 ...')
    print(args)

    ''' 모든 데이터셋에 대해 테스트 '''
    print('\n테스트 시작...')
    with torch.no_grad():
        ''' PSNR/SSIM을 위한 엑셀 파일 생성 '''
        excel_file = ExcelFile()

        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
            excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)

            psnr_epoch_test = float(np.array(psnr_iter_test).mean())
            ssim_epoch_test = float(np.array(ssim_iter_test).mean())
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print('테스트: %s, PSNR/SSIM: %.3f/%.4f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass

        psnr_mean_test = float(np.array(psnr_testset).mean())
        ssim_mean_test = float(np.array(ssim_testset).mean())
        excel_file.add_sheet('ALL', 'Average', psnr_mean_test, ssim_mean_test)
        print('테스트셋 평균 PSNR: %.5f, 평균 SSIM: %.5f' % (psnr_mean_test, ssim_mean_test))
        excel_file.xlsx_file.save(str(result_dir) + '/evaluation.xls')

    pass

if __name__ == '__main__':
    from option import args

    main(args)
