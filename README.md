# Personal Shelf Inspector Training

### Quick Start 
This works locally and in AML as well.
1. Clone repo, change working directory to root of the repo and 

`pip3 install -r requirements.txt`


2. Data for training names and prices are already prepared, [here](https://tkml9458801219.file.core.windows.net/code-391ff5ac-6576-460f-ba4d-7e03433c68b6/Users/karolina.chalupova/personal-shelf-inspector-training/) is the full repo including gitignored files  
You do not have to run `make prepare-data` as the data are already prepared for training names and prices. 

3. Configure training in `Makefile` (see `SETTINGS` and `TRAINING` sections in `Makefile`, also described in README below)
4. Train 
```
make train
```
5. Check inference on testing files
```
make detect
```

### About 
This Repo serves to train detection models for the Personal Shelf Inspector project. The models are then converted to TF.js using [psi-torch-tf-converter](https://github.com/DataSentics/psi-torch-tf-converter) and [psi-tf-js-converter](https://github.com/DataSentics/psi-tf-js-converter) (those will soon be available in containers).

The model architecture ised in this repository is based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5), 
Note that the repositories is a fixed part of this repository (not a repo within repo), specifically a fork of commit xyz FILL IN!!!!!!!!!!!!! with a few bugfixes. Forking this specific commit was necessary for compatibility with psi-torch-tf-conversion.

### Personal Shelf Training Data 
- The training data (jpgs) and trained models are ignored by git, so are not in this remote.
- You can find the entire repo (including the .gitignore-d training images and models) in AML under `Users/karolina.chalupova/personal-shelf-inspector-training` (download [here](https://tkml9458801219.file.core.windows.net/code-391ff5ac-6576-460f-ba4d-7e03433c68b6/Users/karolina.chalupova/personal-shelf-inspector-training/))
- The training data including annotations are also here in Blob Storage `tkml9458801219/Blob Containers/personal-shelf-inspector/`
    - folder `detection_names_and_prices/data/raw` there includes exactly what you need in corresponding folder in this repo (new annotations with decimal values of prices annotated separately)
    - folder `detection_shelves/data/raw` has new annotations of shelves (complete coverage of different shops), but these are now likely redundant considering we do not have to train shelf detection model.
    - we need to add folder `detection_pricetags` similarly, currenlty the data are hidden in the mess of other folders in the repo -- TODO do this when we re-train detection pricetags.

### How to Use This Repo
The data preparation, training, inference and conversion of models to TF.js (mobile models) can be run using `make` commands in the command line. 
In the command line, you need to be in the root of the repo (`cd personal-shelf-inspector-training`)
An example of running a `make` command, which runs scripts that prepare the data for training: 
```  
make prepare-data  
```

For exhaustive list of the commands, and documentation of their settings, refer to the `Makefile`. If you cannot run these commands using `make`, you can execute the commands from the `Makefile` by copying them to the commandline. You can run the commands locally as well as in Azure Machine Learning terminal (using `make` is possible there).

Change `SETTINGS` in the `Makefile` to set up which model and data you want to train (`detection_pricetags` or `detection_names_and_prices` in `FOLDER` argument) and to set up what particular trial run of the training you want to use to train or convert (`TRIAL_NAME`). 


### Data Preparation

1. Gather photos and annotation jsons (VIA format, both polygons and rectangles are acceptable) and save them to the `data/raw` folder in either `detection_pricetags`, `detection_names_and_prices` or `detection_shelves` (depending on what they are used for).
2. To merge all annotation jsons into a single json, convert annotations from VIA format to COCO format, and perform train - validation split, run `make prepare-data` command.
3. Prepare a `settings.yaml` file with paths to training and validation sets, number of classes and their names and save it to the data folder of either `detection_pricetags`, `detection_names_and_prices` or `detection_shelves` (see the yaml in `detection_shelves` for an example).
4. Add `model_settings.yaml` file in the folder `{name of your folder e.g. detection_names_and_prices}/data/models`. Make sure there is the correct number of classes in the model settings. See `detection_names_and_prices/data/models/model_settings.yaml` as an example.

### Model training
Once data preparation is complete just run `make train` command. You can modify and add the training parameters in the `Makefile`.
This will also automatically guide you to optionally setup [Weights and Biases](https://wandb.ai/) to monitor and analze ongoing training.

### Inference 
1. Put testing images to `{detection_shelves, detection_names_and_prices, detection_pricetags}/manual_test_images` folder. 
2. Run `make detect` command. 
Actual inference will be done in javascript.

### Conversion to TF.js

Apply [psi-torch-tf-converter](https://github.com/DataSentics/psi-torch-tf-converter) and [psi-tf-js-converter](https://github.com/DataSentics/psi-tf-js-converter) to the saved torch model.
See the corresponding repositories for more detail
