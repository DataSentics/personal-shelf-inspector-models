SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


# ==================================================================
# SETTINGS  
# ==================================================================


TRIAL_NAME := yolo_640_nano
# - before training a new model, choose any name of the model. 
# - To use a given trained model for inference or to convert the model, specify the model's trial name here.

EXP_NAME := 1
# After training the model, the results are saved to $(FOLDER)/runs/train/$(TRIAL_NAME)_{names_and_prices, pricetags}$EXP_NAME. 
# The exp_name is generated automatically, and starts with "" and continues with increasing numbers 
# To use a given trained model for inference or to convert the model, specify the model's exp name here.

CONDA_ENV_NAME := psi-yolo

setup-dev-env:
	conda env create -f environment.yml ; \
	git clone https://github.com/ultralytics/yolov5 ; \
	($(CONDA_ACTIVATE) $(CONDA_ENV_NAME) ; pip install -r requirements.txt ; cd yolov5 ; bash ./data/scripts/download_weights.sh)

# ==================================================================
# PREPARE TRAINING DATA 
# ==================================================================
prepare-data-names-and-prices:
#   the multiple .json annotations files don't adhere to the same format, so they've already been manually merged and unified into all_annotations.json
# 	python ./utils/merge_jsons.py merge_annotations --path $(FOLDER)/data/raw
	python ./utils/via_to_yolo.py --input_path detection_names_and_prices/data/raw/ --output_path detection_names_and_prices/data/train_test/ --annotations "all_annotations.json" --train_val_split_ratio 0.2
	python ./utils/generate_train_collage.py --dataset_path detection_names_and_prices/data/train_test --output_path detection_names_and_prices/data/train_test_collage --target_size 640 --grid_x 3 --grid_y 7 --n_iterations 50

prepare-data-pricetags:
	echo "prepare-data-pricetags: not implemented yet"

# ==================================================================
# TRAINING 
# ==================================================================

# Runs training. You can alter the training parameters here.  
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

train-pricetags:
	python ./yolov5/train.py \
        --img 640 \
        --batch 16 \
        --epochs 100 \
        --data detection_pricetags/data/settings.yaml \
        --weights ./yolov5/yolov5n.pt \
        --project detection_pricetags/runs/train \
        --name $(TRIAL_NAME)_pricetags \
        --hyp detection_pricetags/models/hyp.finetune.pricetags.640.yaml

# Detection using trained model. Evaluates trained model on provided test data. 
detect-names-and-prices:
	python ./yolov5/detect.py \
        --source ./detection_names_and_prices/manual_test_images \
        --weights ./detection_names_and_prices/runs/train/$(TRIAL_NAME)_names_and_prices${EXP_NAME}/weights/best.pt \
        --conf 0.25 \
        --device cpu


detect-pricetags:
	python ./yolov5/detect.py \
        --source ./detection_pricetags/manual_test_images \
        --weights ./detection_pricetags/runs/train/$(TRIAL_NAME)_pricetags${EXP_NAME}/weights/best.pt \
        --conf 0.25 \
        --device cpu
