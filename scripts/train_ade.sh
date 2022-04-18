set -ex
python train.py --dataroot ./datasets/ade --dataset_mode ade --preprocess none --load_size 1024 --crop_size 768 --name ade_cyclegan --model cycle_gan --pool_size 50 --no_dropout --num_threads 0
python train.py --dataroot ./datasets/ade --dataset_mode ade --preprocess none --load_size 1024 --crop_size 400 --batch_size 2 --name ade_cyclegan --model cycle_gan --pool_size 50 --no_dropout lambda_identity=1.0
