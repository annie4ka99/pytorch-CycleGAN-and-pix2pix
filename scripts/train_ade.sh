set -ex
python train.py --dataroot ./datasets/ade --dataset_mode ade --preprocess none --load_size 1024 --crop_size 1024  --name ade_cyclegan --model cycle_gan --pool_size 50 --no_dropout
