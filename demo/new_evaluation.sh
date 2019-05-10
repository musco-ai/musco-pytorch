#!/bin/bash

echo "Ahoj"
for wf in 0.2 0.3 0.4
do
    python evaluate_demo.py --model resnet50 --dataset imagenet \
                            --data_dir /workspace/raid/data/datasets \
                            --batches_per_train 10000000 \
                            --batches_per_val 10000000 \
                            --batch_size 64 \
                            --save_dir /workspace/raid/data/lmarkeeva/new_exp \
                            --conv_split 1 \
                            --validate_before_ft \
                            --ft_epochs 20 \
                            --patience 3 \
                            --compress_iters 8 \
                            --gpu_number $1 \
                            --weaken_factor $wf

    python evaluate_demo.py --model resnet50 --dataset imagenet \
                            --data_dir /workspace/raid/data/datasets \
                            --batches_per_train 10000000 \
                            --batches_per_val 10000000 \
                            --batch_size 64 \                            
                            --save_dir /workspace/raid/data/lmarkeeva/new_exp_split \
                            --conv_split 1 \
                            --resnet_split \
                            --validate_before_ft \
                            --ft_epochs 20 \
                            --patience 3 \
                            --compress_iters 8 \
                            --gpu_number $1 \
                            --weaken_factor $wf

    python evaluate_demo.py --model resnet18 --dataset imagenet \
                            --data_dir /workspace/raid/data/datasets \
                            --batches_per_train 10000000 \
                            --batches_per_val 10000000 \
                            --batch_size 64 \                            
                            --save_dir /workspace/raid/data/lmarkeeva/new_exp \
                            --conv_split 1 \
                            --validate_before_ft \
                            --ft_epochs 20 \
                            --patience 3 \
                            --compress_iters 8 \
                            --gpu_number $1 \
                            --weaken_factor $wf

    python evaluate_demo.py --model resnet18 --dataset imagenet \
                            --data_dir /workspace/raid/data/datasets \
                            --batches_per_train 10000000 \
                            --batches_per_val 10000000 \
                            --batch_size 64 \                            
                            --save_dir /workspace/raid/data/lmarkeeva/new_exp_split \
                            --conv_split 1 \
                            --resnet_split \
                            --validate_before_ft \
                            --ft_epochs 20 \
                            --patience 3 \
                            --compress_iters 8 \
                            --gpu_number $1 \
                            --weaken_factor $wf

    for split in 1 3 6 13
    do
        python evaluate_demo.py --model vgg16 \
                                --model_weights /workspace/raid/data/eponomarev/pretrained/imagenet/vgg16-397923af.pth \
                                --dataset imagenet \
                                --data_dir /workspace/raid/data/datasets \
                                --batches_per_train 10000000 \
                                --batches_per_val 10000000 \
                                --batch_size 64 \                                
                                --save_dir /workspace/raid/data/lmarkeeva/new_exp \
                                --conv_split $split \
                                --validate_before_ft \
                                --ft_epochs 20 \
                                --patience 3 \
                                --compress_iters 8 \
                                --gpu_number $1 \
                                --weaken_factor $wf
    done
done
