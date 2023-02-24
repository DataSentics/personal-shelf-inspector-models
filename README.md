# Personal Shelf Inspector Models

### Development environment setup
1. Clone the repository, change working directory to the root of the repository

    ```
    git clone git@github.com:DataSentics/psi-torch-models.git
    cd psi-torch-models
    ```

2. Download the training data from the DataSentics sharepoint [here](https://datasentics.sharepoint.com/:u:/s/EXTPersonalShelfInspectorData/ESqfkSgeM_FGr_VZo1UzasgBuu3SDWhXU9MwYI3fcGtcoQ?e=RnZmFT) and unpack
the entire content in the root directory of the repository. You should end up with two new directories `detection_pricetags` and `detection_names_and_prices` in the root of the repository each of which should contain the `raw` subdirectory with image files and an annotation file `annotations.json`. Also, the `detection_pricetags` directory should contain a `settings.yaml` file which contains some basic specifications for model training with yolov5.


3. Setup the development environment by the following Makefile command

    ```
    make venv
    ```

    This will do a couple of things:
        1. A python virtual environment is set up and dependencies needed for `yolov5 v7.0` are installed
        along with optional dependencies which are needed for TF.js model export.
        2. Yolov5 is cloned and copied into the root directory (but only the tag `v7.0` is cloned)
        2. Pretrained model weights for all yolov5 models are downloaded using the `download_weights.sh` script
        provided in the yolov5 repository.


### Data preparation

The Makefile contains targets for data preparation (`prepare-data-names-and-prices`, `prepare-data-pricetags`), which prepare the datasets for training.

This will:

1. split the data into training and validation sets
2. convert the VIA annotations to the annotations format used by yolov5
3. In case of `prepare-data-names-and-prices` it also generates collages of pricetags (see [Collaging]() for more details on this)

### Model training
You can run the training of each model by the following commands. 

```
make train-names-and-prices
make train-pricetags
```

More detailed description of the training procedure and training hyperparameters is described in the Makefile.

### Testing inference
To test your trained models on some photos:
1. Specify the `TRIAL_NAME` and `EXP_NUMBER` of your run in the Makefile (again, for more details checkout the Makefile)
2. Put some test images into `./{detection_names_and_prices, detection_pricetags}/manual_test_images` folder
3. Run the inference

    ```
    make detect-names-and-prices
    make detect-pricetags
    ```

The results are saved to `./{detection_names_and_prices, detection_pricetags}/runs/detect` folder.

### TensorFlow.js export
1. Specify the `TRIAL_NAME` and `EXP_NUMBER` of the trained model (as in inference) in the Makefile
2. Run the export

    ```
    make export-names-and-prices
    make export-pricetags
    ```

    The TensorFlow.js model is exported to `./{detection_names_and_prices, detection_pricetags}/runs/train/${TRIAL_NAME}_{names_and_prices, pricetags}${EXP_NUMBER}/weights/best_web_model` directory.
    For example: `detection_pricetags/runs/train/yolo_640_nano_pricetags5/weights/best_web_model`

### Run demo React web app
You can test the exported TF.js models in a locally-running demo React web application.

```
make run-example-tfjs-webapp
```

