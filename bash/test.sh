# Our model
python test.py --model_name DistgSSR --path_pre_pth ./log/SR_5x5_2x/ALL/DistgSSR/checkpoints/DistgSSR_5x5_2x_epoch_50_model.pth --angRes 5 --scale_factor 2 --data_name ALL

# LLF_InterNet 
python test.py --model_name LF_InterNet --path_pre_pth ./log/SR_5x5_2x/ALL/LF_InterNet/checkpoints/LF_InterNet_5x5_2x_epoch_51_model.pth --angRes 5 --scale_factor 2 --data_name ALL

