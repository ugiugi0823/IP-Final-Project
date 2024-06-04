# Our model
CUDA_VISIBLE_DEVICES=3 python train.py \
--model_name DistgSSR \
--angRes 5 \
--scale_factor 2 \
--batch_size 8



# # LF_InterNet
# CUDA_VISIBLE_DEVICES=3 python train.py \
# --model_name LF_InterNet \
# --angRes 5 \
# --scale_factor 2 \
# --batch_size 8