python train.py \
    --config config_v4.json \
    --input_wavs_dir /root/id-ID-Althaf \
    --input_training_file ID/training.txt \
    --input_validation_file ID/validation.txt \
    --training_epoch 41400 \
    --checkpoint_interval 100000 \
    --validation_interval 10000 \
    --fp16 true --torch_dtype float16