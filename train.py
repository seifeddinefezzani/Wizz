import tensorflow as tf
tf.keras.backend.set_image_data_format("channels_last")
import config
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Input
from tensorflow.keras.utils import pad_sequences
from tqdm import tqdm

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['content'] = df['content'].apply(str).apply(lambda x: x.lower())
    df['content'] = df['content'].apply(lambda x: str(x).replace('\n', ''))  
    df['content'] = df['content'].apply(lambda x: str(x).replace('\t', ''))  
    return df

def get_model_path():
    TIMESTAMP = time.time()
    os.system(f'mkdir model_store/{TIMESTAMP}')
    MODELS_PATH = f'model_store/{TIMESTAMP}/'
    return MODELS_PATH

def prepare_and_save_vocabulary(df: pd.DataFrame) -> dict:
    raw_text = df['content']
    chars = sorted(list(set(''.join(raw_text))))
    char_to_int = dict((c, i+1) for i, c in enumerate(chars))
    int_to_char = dict((i+1, c) for i, c in enumerate(chars))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    with open(config.ENCODING_FILE, 'w') as fp:
        json.dump(char_to_int, fp)
    print("The number of total characters are", n_chars)
    print("\nThe character vocab size is", n_vocab)
    
    return char_to_int

def prepare_input_model(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    train_dataX = []
    train_dataY = []
    valid_dataX = []
    valid_dataY = []

    for i in tqdm(range(len(train_df))):
        train_seq = train_df['content'][i][:config.MAX_SEQUENCE_LENGTH]
        train_dataX.append([char_to_int[char] if char in char_to_int.keys() else len(char_to_int) + 1 for char in train_seq])
        train_dataY.append(train_df['is_sentitive'][i])
    for i in tqdm(range(len(valid_df))):
        valid_seq = valid_df['content'][i][:config.MAX_SEQUENCE_LENGTH]
        valid_dataX.append([char_to_int[char] if char in char_to_int.keys() else len(char_to_int) + 1 for char in valid_seq])
        valid_dataY.append(valid_df['is_sentitive'][i])
    
    X_train = pad_sequences(train_dataX, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post')
    X_valid = pad_sequences(valid_dataX, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post')
    
    y_train = np.array(train_dataY).reshape(-1, 1)
    y_valid = np.array(valid_dataY).reshape(-1, 1)
    
    return X_train, y_train, X_valid, y_valid

def define_model(char_to_int: dict):
    sequence_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(len(char_to_int)+2, # +1 because of 0 padding & +1 because of characters in valid that are not in train 
                                   config.EMBEDDING_DIM,
                                   input_length=config.MAX_SEQUENCE_LENGTH,
                                   name = 'embeddings')
    embedded_sequences = embedding_layer(sequence_input)
    x = LSTM(config.LSTM_UNITS, return_sequences=False, name='lstm_layer')(embedded_sequences)
    x = Dropout(config.DROPOUT)(x)
    x = Dense(config.DENSE_UNITS, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metrics.AUC()])
    return model

def train_model(model, MODELS_PATH: str):
    checkpointer = ModelCheckpoint(filepath=MODELS_PATH + 'best_model.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=1)
    history = model.fit(X_train, y_train, epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, 
                        validation_data=(X_valid, y_valid), verbose=1, callbacks=[checkpointer, reduce_lr])
    return model, history
    
def validate_model(model, MODELS_PATH: str):
    valid_predictions = model.predict(X_valid)
    score = roc_auc_score(y_valid, valid_predictions)
    with open(MODELS_PATH + 'logs.json', 'w') as fp:
        json.dump({'validation score': score}, fp)
    print(f'AUC performance on validation is {score}')
    return valid_predictions

def plot_and_save_losses(history, MODELS_PATH: str):
    plt.figure(figsize=(20, 10))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(MODELS_PATH + 'train_validation_losses.png')
    plt.show();
    
if __name__ == "__main__":

    train_df = pd.read_csv(config.TRAIN_PATH)
    valid_df = pd.read_csv(config.VALID_PATH)
    
    train_df['is_sentitive'] = train_df['is_sentitive'].apply(lambda x: 1 if x else 0)
    valid_df['is_sentitive'] = valid_df['is_sentitive'].apply(lambda x: 1 if x else 0)
    
    train_df = preprocess_data(train_df)
    valid_df = preprocess_data(valid_df)
    char_to_int = prepare_and_save_vocabulary(train_df)
    X_train, y_train, X_valid, y_valid = prepare_input_model(train_df, valid_df)
    MODELS_PATH = get_model_path()
    model = define_model(char_to_int)
    model, history = train_model(model, MODELS_PATH)
    valid_predictions = validate_model(model, MODELS_PATH)
    plot_and_save_losses(history, MODELS_PATH)
                
    