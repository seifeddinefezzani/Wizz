## Paths
WIZZ_DATA_S3 = 's3://bi-optimizedata-test/wizz/'
DATA_PATH = 'data/wizz_dataset.csv'
TRAIN_PATH = 'data/train.csv'
VALID_PATH = 'data/valid.csv'
TEST_PATH = 'data/test.csv'
ENCODING_FILE = 'feature_store/char_to_int.json'
LOG_FILE = 'logs.json'
MODEL_FILE = 'best_model.hdf5'

## Model parameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 50
NUM_EPOCHS = 5
BATCH_SIZE = 20000
LEARNING_RATE = 0.005
LSTM_UNITS = 50
DENSE_UNITS = 30
DROPOUT = 0.1

## Convert to binary
TFLITE = True
TFLITE_MODEL = 'model_store/tflite_wizz.tflite'
COREML = True
COREML_MODEL = 'model_store/coreml_wizz.mlmodel'