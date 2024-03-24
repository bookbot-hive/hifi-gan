python train.py \
    --config config_v4.json \
    --input_wavs_dir EN/audio \
    --input_training_file EN/training.txt \
    --input_validation_file EN/validation.txt \
    --training_epoch 1400 \
    --checkpoint_interval 100000 \
    --validation_interval 10000