import tensorflow as tf
tf.keras.backend.set_image_data_format("channels_last")
import config
import coremltools
import datetime
import glob
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from tqdm import tqdm

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['content'] = df['content'].apply(str).apply(lambda x: x.lower())
    df['content'] = df['content'].apply(lambda x: str(x).replace('\n', ''))  
    df['content'] = df['content'].apply(lambda x: str(x).replace('\t', ''))  
    return df

def prepare_input_model_from_one_sample(text):
    with open(config.ENCODING_FILE, 'r') as fp:
        char_to_int = json.load(fp)
    test_dataX = []
    test_seq = text[:config.MAX_SEQUENCE_LENGTH]
    test_dataX.append([char_to_int[char] if char in char_to_int.keys() else len(char_to_int)+1 for char in test_seq])
    X_test = pad_sequences(test_dataX, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post')
    return X_test
    
def prepare_input_model(df):
    df['content'] = df['content'].apply(str).apply(lambda x: x.lower())
    with open(config.ENCODING_FILE, 'r') as fp:
        char_to_int = json.load(fp)
    test_dataX = []
    for i in range(len(df)):
        test_seq = df['content'][i][:config.MAX_SEQUENCE_LENGTH]
        test_dataX.append([char_to_int[char] if char in char_to_int.keys() else len(char_to_int)+1 for char in test_seq])
    X_test = pad_sequences(test_dataX, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post')
    return X_test

def load_best_model():
    score = -1
    for folder in os.listdir('model_store/'):
        try:
            with open('model_store/' + folder + '/' + config.LOG_FILE, 'r') as fp:
                score_auc = json.load(fp)['validation score']
                if score_auc > score:
                    score = score_auc
                    best_model_path = 'model_store/' + folder + '/' + config.MODEL_FILE
        except:
            pass
    
    model = load_model(best_model_path)
    return model, best_model_path

if __name__ == "__main__":

    test_df = pd.read_csv(config.TEST_PATH)
    test_df['is_sentitive'] = test_df['is_sentitive'].apply(lambda x: 1 if x else 0)
    model, best_model_path = load_best_model()
    
    if config.COREML:
        coreml_model = coremltools.converters.convert(best_model_path)
        coreml_model.save(config.COREML_MODEL)
        
    if config.TFLITE:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
          tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()
        with open(config.TFLITE_MODEL, 'wb') as f:
            f.write(tflite_model)
    
    with open(config.ENCODING_FILE, 'r') as fp:
        char_to_int = json.load(fp)
    
    test_df = preprocess_data(test_df)
    X_test = prepare_input_model(test_df)
    y_test = np.array(test_df['is_sentitive']).reshape(-1, 1)
    test_predictions = model.predict(X_test)
    print(f'AUC performance on all test is {roc_auc_score(y_test, test_predictions)}')
