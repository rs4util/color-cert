#!/bin/bash

# Train the model
python color_cert.py --dataset cifar10 --ckptdir ckpt \
                     --min_sigma 0.25 --max_sigma 0.5 --num_img 500 --batch_size 128 --epochs 300 --lr 0.01 \
                     --gauss_num 16 --lbd1 6.0 --lbd2 1.0 --gamma 8.0