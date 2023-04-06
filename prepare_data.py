import config
import gzip
import os 
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare_folders():
    elements = os.listdir()
    for folder in ['feature_store', 'model_store']:
        if folder not in elements:
            os.system(f'mkdir {folder}')
            
def download_data_from_s3():
    os.system(f'aws s3 cp {config.WIZZ_DATA_S3} . --recursive')
    
def prepare_data():
    dataset = pd.DataFrame()
    for date_folder in os.listdir('data'):
        try:
            for file_name in tqdm(os.listdir('data/' + date_folder)):
                file_path = 'data/' + date_folder + '/' + file_name
                with gzip.open(file_path) as f:
                    data = f.read().decode("utf-8")
                df = pd.DataFrame(data.split('N\n'))[0].str.split(',', expand=True)
                one_file_df = pd.DataFrame()
                one_file_df[['user_id', 'discussion_id', 'type', 'placement', 'content', 'is_sentitive']] = df[df.columns[:6]]
                one_file_df['date'] = date_folder
                dataset = pd.concat([dataset, one_file_df])
        except:
            print(f'{date_folder} is not a folder')
    dataset = dataset[dataset['is_sentitive'].isin(['true', 'false'])].reset_index(drop=True)
    return dataset

def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, valid_df, test_df

if __name__ == "__main__":

    prepare_folders()
    download_data_from_s3()
    dataset = prepare_data()
    train_df, valid_df, test_df = split_data(dataset)
    dataset.to_csv(config.DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    valid_df.to_csv(config.VALID_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)