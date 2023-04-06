### Requirements
To be able to run the code in python and to generate the models in coreML & tensorflow lite, you will need to run the commands below, it will create a conda environnement using python v3.8 and it will install the needed libraries

```bash
conda create --name wizz python=3.8
```

```bash
pip install -r requirements.txt
```

```bash
source activate wizz
```

### Configs
There is a file config.py where you can find the modeling parameters and the local paths you will be using
Note that you will probably need to change the S3 path of the data you sent us (we copied your data in bi-optimizedata-test), maybe you don't have access to this bucket

### Code instructions
Once you created your environnement here are three files you need to run:

```bash
- python prepare_data.py
```

-It will:
    - Create three folders
        - data/ where you will find the downloaded data from S3
        - an empty folder: feature_store/ where you will find later a json file called chars_to_ids that maps all possible characters to numerical ids
        - an empty folder: model_store/ where you will find later all the models you trained with the validation scores and the train/validation losses
    - Split the data into train & validation & test datasets for modeling

```bash
- python train.py
```

It will:
    - generate the file chars_to_ids and save it in the folder feature_store/
    - train an LSTM model at character level, once the training is finished it will save the model and the logs in the folder model_store/
    - test the model on the validation set (which is used for model tuning) and compute the area under curve (AUC) score which is the metric we use to evaluate the performance of a model when the target is highly imbalanced (around 98% not toxic & 2% toxic)

```bash
- python test.py
```

It will:
    - load the file chars_to_ids from the folder feature_store/
    - load the best model trained from the folder model_store
    - test the best model on the testing set
    - convert the model:
        - for android using tensorflow lite
        - for ios using coreML
    - save the converted models in the folder model_store/
    
### Tune the model
For now, you can tune the model and restart the training using the file config.py - you will find the model parameters
The best model is the one that gives the highest validation AUC score.

### How to use the packaged model
There are two steps you need to do to be able to use the packaged model
    - First, you need to preprocess the text. You will find all the preprocessing done in the function preprare_data/ in train.py & test.py, the preprocessing is easy to code in any programming language
    - Then, you need to load the json file and to map the message (after preprocessing) from chars to integers
    - Every message should have the size (1, config.MAX_SEQUENCE_LENGTH)
        - If length(message) > MAX_SEQUENCE_LENGTH: message = message[:MAX_SEQUENCE_LENGTH]
        - If length(message) < MAX_SEQUENCE_LENGTH: message = message + 0 padding (add 0 until length(message) = MAX_SEQUENCE_LENGTH)
        - The input of the model should be 2D (NUM_SAMPLES,  config.MAX_SEQUENCE_LENGTH)