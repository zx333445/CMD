#!/bin/bash
# MODEL_NAME='resnet50'
MODEL_NAME='densenet169'


# SAVE_MODEL_PATH="./results/models/cmd_resnet50.pth"
# SAVE_MODEL_PATH="./results/models/cmd_densenet169.pth"
SAVE_MODEL_PATH="./results/models/cmd_swinb.pth"
# SAVE_MODEL_PATH="./results/models/cmd_convnext.pth"


# SAVE_MODEL_PATH="./results/models/cas_resnet50.pth"
# SAVE_MODEL_PATH="./results/models/cas_densenet169.pth"
# SAVE_MODEL_PATH="./results/models/cas_swinb.pth"
# SAVE_MODEL_PATH="./results/models/cas_convb.pth"


# LOG_DIR="./logs/cmd_densenet169"
LOG_DIR="./logs/cmd_swinb"
# LOG_DIR="./logs/cmd_convnext"
# LOG_DIR="./logs/cmd_resnet50"


# LOG_DIR="./logs/cas_resnet50"
# LOG_DIR="./logs/cas_densenet169"
# LOG_DIR="./logs/cas_swinb"
# LOG_DIR="./logs/cas_convb"

NUM_EPOCHS=50


python train.py \
    --model_name $MODEL_NAME \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --log_dir $LOG_DIR \
    SGD
