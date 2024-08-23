# 1814 data / 16 bs = 113 steps/ep
# 2.4M steps / 113 steps/ep = 21238 eps
# 200k steps / 113 steps/ep = 1770 eps

python fine_tune.py \
    --config config_v4.json \
    --input_wavs_dir /home/s44504/sw-TZ-Victoria/sw-TZ-Victoria/ \
    --input_training_file SW/training.txt \
    --input_validation_file SW/validation.txt \
    --training_epoch 1770 \
    --checkpoint_interval 100000 \
    --validation_interval 10000 \
    --fp16 true --torch_dtype bfloat16