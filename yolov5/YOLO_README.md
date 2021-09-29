Majority of repo comes from https://github.com/ultralytics/yolov5. Few modifications are introduced and they are described below. The rest comes from the original reposiory.

## Installation
Follow original instructions, no problem was observed.

## Converting dataset to COCO format
To convert dataset with annotations suitable for our Mask RCNN (i.e. with via_region_data.json),
use `via_to_coco.py`. The scripts take as input path to folder with train and val data.
Each folder should contain photos and annotations named via_region_data.json.
The last argument specifies a folder, where the processed photos and annotations will be saved.
For more info about the output structure and format, see https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data.

You can use bash script `prepare_data.sh` with prefilled arguments by calling `bash prepare_data.sh`.

## Training 
To train model on your data, follow https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data.
The training can start by running e.g. `bash train_fridge.sh`, for change or more info, look inside. To train on another dataset, proceed according to section above and modify the shell script, see e.g. `train_pricetags.sh`. Make sure the .yaml file given by the --data parameter of the shell scripts contains the right paths to images.

After the training begins, a new folder is created in `runs/train` containing logs and weights.
Also, it contains image with ground truth labels and prediciton from the beginning.
It can be useful to check the correctness of transformed coordinates.

## Prediction
Use e.g. `detect.sh`, which calls original script `detect.py` with prefilled arguments.

## Model wrapper
Wrapper of trained yolov5 model is introduced in `model.py`. It includes definition of basic `Model` class with methods `load` and `detect` for easier manipulation with trained model, similarly as with Mask R-CNN which we use on multiple projects. The `detect` method specifiec the output format of model detections. It can be used to detect on photos, video detection requires further modifications. The notebook `model.ipynb` is only for testing the `Model` class in JupyterLab.


## Trained model weights
Weights for few trained models are stored in `weights` folder to be used later. Weights for pretrained yolov5 small are in `yolo5s.pt`.
