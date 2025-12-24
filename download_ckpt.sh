#!/bin/bash

echo "Downloading matting checkpoint ..."
gdown "https://drive.google.com/uc?id=1xzUPTKCC67CUnBx6T4xPwN8aGAB92YKX" -O ./src/layerd/weight/matting_model.pth

echo "Downloading hi-sam decoder checkpoint ..."
gdown "https://drive.google.com/uc?id=1GQJTLNY_W3QPz8ZE9YsZb9BAL9Wgtd2G" -O ./hi_sam/pretrained_checkpoint/efficient_hi_sam_s.pth

echo "Downloading SAM encoder checkpoint ..."
wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -O ./hi_sam/pretrained_checkpoint/sam_vit_h_4b8939.pth

echo "All checkpoints downloaded successfully."