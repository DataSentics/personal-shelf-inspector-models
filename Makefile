SHELL=/bin/bash

# ==================================================================
# SETTINGS  
# ==================================================================
FOLDER := detection_names_and_prices
# Specify what model you are training.
# Choose between detection_pricetags and detection_names_and_prices.
# This serves to point to appropriate model folder in the root containing training data, models, etc.

TRIAL_NAME := basic_320
# - before training a new model, choose any name of the model. 
# - To use a given trained model for inference or to convert the model, specify the model's trial name here.
# current best trial names: 
# shelves: r320_e200_b32_oldyolo
# pricetags: basic_320
# names_and_prices: TODO

EXP_NAME := exp1
# After training the model, the results are saved to $(FOLDER)/runs/train/$(EXP_NAME)_$(TRIAL_NAME). 
# The exp_name is generated automatically, and starts with exp and continues with number, e.g., exp1 
# To use a given trained model for inference or to convert the model, specify the model's exp name here.


# ==================================================================
# PREPARE TRAINING DATA 
# ==================================================================
prepare-data: 
	python ./utils/merge_jsons.py merge_annotations \
	    --path $(FOLDER)/data/raw
	python ./utils/via_to_coco.py \
	    --input_path_train $(FOLDER)/data/raw/ \
	    --output_path $(FOLDER)/data/train_test/ \
	    --ann_type rectangle \
	    --annotations "all_annotations.json"

# ==================================================================
# TRAINING 
# ==================================================================

# Runs training. You can alter the training parameters here.  
# You may pass more training parameters, see ./yolov5/train.py for the exhaustive list. 
train:
    python ./yolov5/train.py \
        --img 320 \
        --batch 32 \
        --epochs 1 \
        --data $(FOLDER)/data/settings.yaml \
        --weights ./yolov5/weights/yolov5s.pt \
        --logdir $(FOLDER)/runs/train \
        --name $(TRIAL_NAME) \
        --hyp ./yolov5/data/hyp.scratch.yaml

# Resumes the last training where it left of. 
resume: 
    python ./yolov5/train.py --resume

# Detection using trained model. Evaluates trained model on provided test data. 
detect: 
    python ./yolov5/detect.py \
        --source ./$(FOLDER)/manual_test_images \
        --weights ./$(FOLDER)/runs/train/$(EXP_NAME)_$(TRIAL_NAME)/weights/best.pt \
        --conf 0.25 \
        --device cpu
