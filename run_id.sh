# 929 data / 16 bs = 58 steps/ep
# 2.4M steps / 58 steps/ep = 41379 eps
# 200k steps / 58 steps/ep = 3500 eps

# python train.py \
#     --config config_v4.json \
#     --input_wavs_dir /root/id-ID-Althaf \
#     --input_training_file ID/training.txt \
#     --input_validation_file ID/validation.txt \
#     --training_epoch 41400 \
#     --checkpoint_interval 100000 \
#     --validation_interval 10000

python fine_tune.py \
    --config config_v4.json \
    --input_wavs_dir /root/id-ID-Althaf \
    --input_training_file ID/training.txt \
    --input_validation_file ID/validation.txt \
    --training_epoch 3500 \
    --checkpoint_interval 100000 \
    --validation_interval 10000 \
    --fp16 true --torch_dtype bfloat16