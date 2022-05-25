#!/bin/bash

# Test the model
python color_cert.py --task test --num_img 500 --batch_size 32\
                     --min_sigma 0.25 --max_sigma 0.5 \
                     --resume_f_ckpt path/to/classifeir/checkpoint \
                     --resume_g_ckpt path/to/generator/checkpoint