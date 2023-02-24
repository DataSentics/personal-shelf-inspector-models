SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: help venv prepare-data-pricetags prepare-data-pricetags prepare-data-names-and-prices train-pricetags detect-names-and-prices detect-pricetags export-names-and-prices export-pricetags run-example-tfjs-webapp

# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-z%A-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ==================================================================
# TRIAL NAME SETTINGS
# ==================================================================

# Before training a new model, pick a trial name for the experiment
# Use the same trial name for the inference with manual_test_images or for the export to TF.js
TRIAL_NAME := yolo_640_nano

# When training the model, the results (weights, tensorboard logs, metrics) are stored in:
# 	./{detection_names_and_prices, detection_pricetags}/runs/train/$(TRIAL_NAME)_{names_and_prices, pricetags}$EXP_NUMBER. 
#
# The EXP_NUMBER is generated automatically by the YOLOv5's train.py script 
# 	- It starts with "" and is incremented by 1 for each new trial with the same $TRIAL_NAME, such that the subsequent runs are not overwritten.
# 
# Make sure to specify the correct $EXP_NUMBER when running the inference or when exporting to TF.js.

# best names-and-prices model
EXP_NUMBER_NAMES_PRICES := 2

# best pricetags model
EXP_NUMBER_PRICETAGS := 5


CONDA_ENV_NAME := psi-yolo


# ==================================================================
# ENVIRONMENT SETUP
# ==================================================================

venv: ## set up a simple python virtual env, clone yolov5 version 7.0 and get model weights
	python3 -m venv .venv \
	&& source .venv/bin/activate \
	&& python3 -m pip install --upgrade pip \
	&& python3 -m pip install -r requirements.txt \
  	&& git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5 \
 	&& cd yolov5 \
  	&& bash ./data/scripts/download_weights.sh


# ==================================================================
# PREPARE TRAINING DATA 
# ==================================================================

prepare-data-pricetags:

	python ./utils/via_to_yolo.py --input_path detection_pricetags/data/raw/ --output_path detection_pricetags/data/train_test/ --annotations "annotations.json" --train_val_split_ratio 0.2

prepare-data-names-and-prices:

#   Photos and VIA (https://www.robots.ox.ac.uk/~vgg/software/via/via.html) generated annotations for training are stored in the detection_names_and_prices/data/raw/ directory
#   via_to_yolo.py converts the VIA json annotations to the YOLO format and produces a train-test split.
#   generate_train_collage.py generates pricetag collages and saves them into train_test_collage folder. This is used for training.
#
	python ./utils/via_to_yolo.py --input_path detection_names_and_prices/data/raw/ --output_path detection_names_and_prices/data/train_test/ --annotations "annotations.json" --train_val_split_ratio 0.2
	python ./utils/generate_train_collage.py --dataset_path detection_names_and_prices/data/train_test --output_path detection_names_and_prices/data/train_test_collage --target_size 640 --grid_x 3 --grid_y 7 --n_iterations 50

# ==================================================================
# TRAINING
# ==================================================================

# You can alter some training parameters here - YOLOv5 size (nano, small, medium..), image size, batch size, number of epochs, etc.
# the remaining hyperparameters are specified in <detection_pricetags/models/hyp.finetune.pricetags.640.yaml> for the pricetag detection model
# and in <detection_names_and_prices/models/hyp.finetune.collage.640.yaml> for the names and prices detection model


train-pricetags: ## runs the training script for the model which detects pricetags in images of shelves
	python ./yolov5/train.py \
        --img 640 \
        --batch 48 \
        --epochs 300 \
        --data detection_pricetags/settings.yaml \
        --weights ./yolov5/weights/yolov5n.pt \
        --project detection_pricetags/runs/train \
        --name $(TRIAL_NAME)_pricetags \
        --hyp detection_pricetags/models/hyp.finetune.pricetags.640.yaml

# Runs training.  file
# You may pass more training parameters, see ./yolov5/train.py for the exhaustive list. 
train-names-and-prices: ## runs the training script for the model which detects names and prices in images of pricetags
	python ./yolov5/train.py \
        --img 640 \
        --batch 48 \
        --epochs 300 \
        --data detection_names_and_prices/data/train_test_collage/settings.yaml \
        --weights ./yolov5/weights/yolov5n.pt \
        --project detection_names_and_prices/runs/train \
        --name $(TRIAL_NAME)_names_and_prices \
        --hyp detection_names_and_prices/models/hyp.finetune.collage.640.yaml


# ==================================================================
# INFERENCE/TESTING
# ==================================================================
# This is mostly for development and testing purposes


# Detection using trained model on manual_test_images.
detect-names-and-prices:
	python ./yolov5/detect.py \
        --source ./detection_names_and_prices/manual_test_images \
        --weights ./detection_names_and_prices/runs/train/$(TRIAL_NAME)_names_and_prices${EXP_NUMBER_NAMES_PRICES}/weights/best.pt \
		--project ./detection_names_and_prices/runs/detect \
		--name $(TRIAL_NAME)_names_and_prices${EXP_NUMBER_NAMES_PRICES} \
        --conf 0.25 \
        --device cpu


detect-pricetags:
	python ./yolov5/detect.py \
        --source ./detection_pricetags/manual_test_images \
        --weights ./detection_pricetags/runs/train/$(TRIAL_NAME)_pricetags${EXP_NUMBER_PRICETAGS}/weights/best.pt \
		--project ./detection_pricetags/runs/detect \
		--name $(TRIAL_NAME)_pricetags${EXP_NUMBER_PRICETAGS} \
        --conf 0.25 \
		--device cpu

# ==================================================================
# EXPORT TO BINARY
# ==================================================================
# Export the model on CPU (GPU export can crash when not enough GPU memory is available)
# to hide the GPU from tensorflow -> export CUDA_VISIBLE_DEVICES=''
export-names-and-prices:
	export CUDA_VISIBLE_DEVICES='' ; \
	python ./yolov5/export.py \
		--weights ./detection_names_and_prices/runs/train/$(TRIAL_NAME)_names_and_prices${EXP_NUMBER_NAMES_PRICES}/weights/best.pt \
		--include tfjs saved_model

export-pricetags:
	export CUDA_VISIBLE_DEVICES='' ; \
	python ./yolov5/export.py \
		--weights ./detection_pricetags/runs/train/$(TRIAL_NAME)_pricetags${EXP_NUMBER_PRICETAGS}/weights/best.pt \
		--include tfjs saved_model


# Run demo React web app that uses the exported model (to check that the TF.js export worked)
# if the export to TF.js was successful, the model should be loadable in the web browser and we can use it for inference.
run-example-tfjs-webapp:
	git clone https://github.com/zldrobit/tfjs-yolov5-example.git ; \
	cd tfjs-yolov5-example \
		&& npm install \
		&& ln -f -s ../../detection_names_and_prices/runs/train/$(TRIAL_NAME)_names_and_prices${EXP_NUMBER_NAMES_PRICES}/weights/best_web_model public/web_model \
		&& npm start

