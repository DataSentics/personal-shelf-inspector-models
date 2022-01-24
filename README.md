# Personal Shelf Inspector Training

### Setup
1. Clone repo, change working directory to root of the repo

```
$ git clone git@github.com:DataSentics/psi-torch-models.git
$ cd psi-torch-models
```

2. Add [DVC](https://dvc.org/) Azure Blob storage credentials to `.dvc/config.local`. There is a template `.dvc/config.local.example` for inspiration.
The DVC remote contains the datasets, trained models and training runs artifacts, which are too big to store in github repo.


```
file: .dvc/config.local.example
---

['remote "ps_azure"']
    account_name = <your account name>
    account_key = <your account key>
```

3. Setup the conda development environment. Make sure conda is installed on your system. 

```
$ make setup-dev-env
$ conda activate psi-yolo
```

### Training
1. Prepare the data (optional)

The Makefile contains targets for data preparation (`prepare-data-names-and-prices`, `prepare-data-pricetags`), which prepare the datasets for training.

But the prepared versions of thoses datasets are already stored in DVC remote, so run `make prepare-data-{names-and-prices, pricetags}` only if you make any changes to the source datasets.

2. Run the training

```
$ make train-names-and-prices
$ make train-pricetags
```

More detailed description of the training procedure and training hyperparameters is described in the Makefile.

The training script will also automatically guide you to optionally setup [Weights and Biases](https://wandb.ai/) to monitor and analze ongoing training.


### Inference
To test your trained models on some photos:
1. Specify the `TRIAL_NAME` and `EXP_NUMBER` of your run in the Makefile (again, for more details checkout the Makefile)
2. Put some test images into `./{detection_names_and_prices, detection_pricetags}/manual_test_images` folder
3. Run the inference

```
$ make detect-names-and-prices
$ make detect-pricetags
```

The results are saved to `./{detection_names_and_prices, detection_pricetags}/runs/detect` folder.

### TensorFlow.js export
1. Specify the `TRIAL_NAME` and `EXP_NUMBER` of the trained model (as in inference) in the Makefile
2. Run the export

```
$ make export-names-and-prices
$ make export-pricetags
```

The TensorFlow.js model is exported to `./{detection_names_and_prices, detection_pricetags}/runs/train/${TRIAL_NAME}_{names_and_prices, pricetags}${EXP_NUMBER}/weights/best_web_model` directory.
For example: `detection_pricetags/runs/train/yolo_640_nano_pricetags5/weights/best_web_model`

### Run demo React web app
You can test the exported TF.js models in a locally-running demo React web application.

```
$ make run-example-tfjs-webapp
```

