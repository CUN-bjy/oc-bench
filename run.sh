CUDA_VISIBLE_DEVICES=4,5 python scripts/train_steve.py --image_size 128 --batch_size 24 --num_workers 8 --prefix steve-clevr-n5 --num_slots 5 --seed 0
CUDA_VISIBLE_DEVICES=4,5 python scripts/train_steve.py --image_size 128 --batch_size 24 --num_workers 8 --prefix steve-clevr-n10 --num_slots 10 --seed 0
CUDA_VISIBLE_DEVICES=4,5 python scripts/train_steve.py --image_size 128 --batch_size 24 --num_workers 8 --prefix steve-clevr-n15 --num_slots 15 --seed 0
