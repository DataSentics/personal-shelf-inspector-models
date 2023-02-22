SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: setup-dev-env download-data prepare-data-pricetags train-names-and-prices train-pricetags detect-names-and-prices detect-pricetags export-names-and-prices export-pricetags run-example-tfjs-webapp

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

# First make sure you have conda installed.
#
# We first install mamba into the conda base environment.
# 	- mamba is just a wrapper around conda with better and faster dependency resolution
#
# Then we create a conda environment from the environment.yml file, which specifies compatible pytorch, tensorflow, python, cudnn, and cudatoolkit versions.
# 	- We don't install these dependencies with pip, because pip cannot ensure the binary compatibility of pytorch and tensorflow with cuda.
#
# The rest of the dependencies (mainly standard python packages) are installed with pip in the conda environment.
#
# Then we download the latest YOLOv5 library along with pretrained network weights.
#
# And finally we download the data for the training and inference from data version control remote (dvc pull)
#
# After the successful environment setup, activate the conda environment (conda activate psi-yolo) and you should be able to run all the other Makefile targets.
setup-dev-env:
	($(CONDA_ACTIVATE) base ; \
		conda install mamba -n base -c conda-forge ; \
		mamba env create -f environment.yml ; \
		conda activate $(CONDA_ENV_NAME) ; \
		pip install -r requirements.txt ; \
		git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5 ; \
		cd yolov5 ; \
		bash ./data/scripts/download_weights.sh \
	)
	@echo -e "\n\n\nSetup environment completed. Pulling data from DVC remote...\n"
	($(CONDA_ACTIVATE) $(CONDA_ENV_NAME) ; \
		dvc pull \
	)

# ==================================================================
# PREPARE TRAINING DATA 
# ==================================================================

# Pull the datasets, trained models and experiment runs artifacts from dataversion control (dvc) remote (Azure Blob Storage).
# Make sure you have valid credentials for the remote (Azure Blob Storage) in the .dvc/config.local file.
# If not, checkout the .dvc/config.local.example template, fill in your credentials and rename it to .dvc/config.local
download-data:
	dvc pull

prepare-data-names-and-prices:
#   the multiple .json annotation files don't adhere to the same format, so they've already been manually merged and unified into all_annotations.json
#   -> so there is no need to merge JSONs with merge_jsons.py
# 
#   via_to_yolo.py converts the VIA json annotations to the YOLO format and produces train_test split
#   generate_train_collage.py generates pricetag collages and saves them into train_test_collage folder. This is used for training.
#
# 	python ./utils/merge_jsons.py merge_annotations --path $(FOLDER)/data/raw
	python ./utils/via_to_yolo.py --input_path detection_names_and_prices/data/raw/ --output_path detection_names_and_prices/data/train_test/ --annotations "all_annotations.json" --train_val_split_ratio 0.2
	python ./utils/generate_train_collage.py --dataset_path detection_names_and_prices/data/train_test --output_path detection_names_and_prices/data/train_test_collage --target_size 640 --grid_x 3 --grid_y 7 --n_iterations 50

prepare-data-pricetags:
	@echo -e "prepare-data-pricetags:\nthe original pricetag training dataset was lost and only the train_test split was available - so we don't need to convert the VIA json annotations to the YOLO format"
	@echo "  -> the already-prepared YOLO pricetag dataset is just pulled from the DVC repo"

# ==================================================================
# TRAINING
# ==================================================================

# Runs training. You can alter some training parameters here - YOLOv5 size (nano, small, medium..), image size, batch size, number of epochs, etc.
# the remaining hyperparameters are specified in detection_names_and_prices/models/hyp.finetune.collage.640.yaml file
# You may pass more training parameters, see ./yolov5/train.py for the exhaustive list. 
train-names-and-prices:
	python ./yolov5/train.py \
        --img 640 \
        --batch 48 \
        --epochs 300 \
        --data detection_names_and_prices/data/train_test_collage/settings.yaml \
        --weights ./yolov5/yolov5n.pt \
        --project detection_names_and_prices/runs/train \
        --name $(TRIAL_NAME)_names_and_prices \
        --hyp detection_names_and_prices/models/hyp.finetune.collage.640.yaml

# pricetags are trained with different hyperparameters
# -> see detection_pricetags/models/hyp.finetune.pricetags.640.yaml
train-pricetags:
	python ./yolov5/train.py \
        --img 640 \
        --batch 48 \
        --epochs 300 \
        --data detection_pricetags/data/settings.yaml \
        --weights ./yolov5/yolov5n.pt \
        --project detection_pricetags/runs/train \
        --name $(TRIAL_NAME)_pricetags \
        --hyp detection_pricetags/models/hyp.finetune.pricetags.640.yaml

# ==================================================================
# INFERENCE
# ==================================================================

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

