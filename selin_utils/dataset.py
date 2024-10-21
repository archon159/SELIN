"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The module that contains utility functions and classes related to datasets
"""
from typing import Dict, List, Tuple
import math
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import random

from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import get_image_backend
import torchvision.transforms as transforms

NLP_DATASET = ['yelp', 'clickbait']
NLP_BASE = ['bert', 'roberta', 'llama2']
TAB_DATASET = ['adult', 'diabetes', 'credit']
TAB_BASE = ['dnn', 'fttransformer']

class YelpDataset(Dataset):
    """
    Dataset structure for Yelp data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        is_train: bool=True,
        atom_pool: object=None,
        atom_tokenizer: object=None,
        tf_tokenizer: object=None,
        max_len: int=512,
        noise_ratio: float=0.,
    ):
        self.text = data_df['text']
        self.y = data_df['label'].astype(dtype='int64')
        
        if noise_ratio > 0:
            noise_y = []
            original_y = self.y.tolist()
            total_num = len(original_y)
            idx = list(range(total_num))
            random.shuffle(idx)
            num_noise = int(noise_ratio * total_num)            
            noise_idx = idx[:num_noise]
            for i in range(total_num):
                if i in noise_idx:
                    noiselabel = 1 - original_y[i] # flipping
                    noise_y.append(int(noiselabel))
                else:    
                    noise_y.append(int(original_y[i]))  
            self.y = noise_y
            
        self.atom_pool = atom_pool
        self.atom_tokenizer = atom_tokenizer
        self.tf_tokenizer = tf_tokenizer
        self.max_len = max_len
        
        self.is_train = is_train

    def __len__(
        self
    ) -> int:
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        if self.atom_pool != None:
            text_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count = Counter(self.atom_tokenizer.tokenize(self.text[i]))

            for word, count in dict(x_count).items():
                text_[word] = count

            # x_ indicates if the satisfaction of atoms for current sample
            x_ = self.atom_pool.check_atoms(text_)

            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
            
        y = self.y.iloc[i]
        
        encoding = self.tf_tokenizer.encode_plus(
            text=self.text[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = input_ids.squeeze(dim=0).long()
        attention_mask = attention_mask.squeeze(dim=0).long()

        return (input_ids, attention_mask, x_), y

class ClickbaitDataset(Dataset):
    """
    Dataset structure for clickbait data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        is_train: bool,
        atom_pool: object,
        atom_tokenizer: object,
        tf_tokenizer: object,
        max_len: int=512,
        noise_ratio: float=0.,
    ):
        self.title = data_df['title']
        self.text = data_df['text']
        self.y = data_df['label'].astype(dtype='int64')
        
        if noise_ratio > 0:
            noise_y = []
            original_y = self.y.tolist()
            total_num = len(original_y)
            idx = list(range(total_num))
            random.shuffle(idx)
            num_noise = int(noise_ratio * total_num)            
            noise_idx = idx[:num_noise]
            for i in range(total_num):
                if i in noise_idx:
                    noiselabel = 1 - original_y[i] # flipping
                    noise_y.append(int(noiselabel))
                else:    
                    noise_y.append(int(original_y[i]))  
            self.y = noise_y
        
        self.atom_pool = atom_pool
        self.tf_tokenizer = tf_tokenizer
        self.atom_tokenizer = atom_tokenizer
        self.max_len = max_len
        
        self.is_train = is_train

        print(f"Data Num: {len(self.y)}")

    def __len__(
        self
    ) -> int:
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        if self.atom_pool != None:
            title_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count_title = Counter(self.atom_tokenizer.tokenize(self.title[i]))

            for word, count in dict(x_count_title).items():
                title_[word] = count

            text_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count_text = Counter(self.atom_tokenizer.tokenize(self.text[i]))

            for word, count in dict(x_count_text).items():
                text_[word] = count

            article_ = np.concatenate((title_, text_))

            x_ = self.atom_pool.check_atoms(article_)
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        y = self.y.iloc[i]
        
        encoding = self.tf_tokenizer.encode_plus(
            text=self.title[i],
            text_pair=self.text[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = input_ids.squeeze(dim=0).long()
        attention_mask = attention_mask.squeeze(dim=0).long()

        return (input_ids, attention_mask, x_), y
    
class AdultDataset(Dataset):
    """
    Dataset structure for adult data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        is_train: bool,
        atom_pool: object,
        noise_ratio: float=0.,
    ):
        self.x = data_df.drop(columns=['income'])
        self.y = data_df['income'].astype(dtype='int64')
        
        if noise_ratio > 0:
            noise_y = []
            original_y = self.y.tolist()
            total_num = len(original_y)
            idx = list(range(total_num))
            random.shuffle(idx)
            num_noise = int(noise_ratio * total_num)            
            noise_idx = idx[:num_noise]
            for i in range(total_num):
                if i in noise_idx:
                    noiselabel = 1 - original_y[i] # flipping
                    noise_y.append(int(noiselabel))
                else:    
                    noise_y.append(int(original_y[i]))  
            self.y = noise_y
        
        self.atom_pool = atom_pool
        
        self.is_train = is_train

        print(f"Data Num: {len(self.y)}")

    def __len__(
        self
    ):
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        x_dummy = self.x.loc[i]

        if self.atom_pool != None:
            x_ = self.atom_pool.check_atoms(np.array(x_dummy))
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        y = self.y[i]
        
        

        x = torch.Tensor(x_dummy).float()
        return (x, x_), y

class DiabetesDataset(Dataset):
    """
    Dataset structure for adult data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        is_train: bool,
        atom_pool: object,
        noise_ratio: float=0.,
    ):
        self.x = data_df.drop(columns=['Diabetes_binary'])
        self.y = data_df['Diabetes_binary'].astype(dtype='int64')
        
        if noise_ratio > 0:
            noise_y = []
            original_y = self.y.tolist()
            total_num = len(original_y)
            idx = list(range(total_num))
            random.shuffle(idx)
            num_noise = int(noise_ratio * total_num)            
            noise_idx = idx[:num_noise]
            for i in range(total_num):
                if i in noise_idx:
                    noiselabel = 1 - original_y[i] # flipping
                    noise_y.append(int(noiselabel))
                else:    
                    noise_y.append(int(original_y[i]))  
            self.y = noise_y
        
        self.atom_pool = atom_pool
        
        self.is_train = is_train

        print(f"Data Num: {len(self.y)}")

    def __len__(
        self
    ):
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        x_dummy = self.x.loc[i]

        if self.atom_pool != None:
            x_ = self.atom_pool.check_atoms(np.array(x_dummy))
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        y = self.y[i]
        
        x = torch.Tensor(x_dummy).float()
        return (x, x_), y
    
class CreditDataset(Dataset):
    """
    Dataset structure for adult data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        is_train: bool,
        atom_pool: object,
        noise_ratio: float=0.,
    ):
        self.x = data_df.drop(columns=['default.payment.next.month'])
        self.y = data_df['default.payment.next.month'].astype(dtype='int64')
        
        if noise_ratio > 0:
            noise_y = []
            original_y = self.y.tolist()
            total_num = len(original_y)
            idx = list(range(total_num))
            random.shuffle(idx)
            num_noise = int(noise_ratio * total_num)            
            noise_idx = idx[:num_noise]
            for i in range(total_num):
                if i in noise_idx:
                    noiselabel = 1 - original_y[i] # flipping
                    noise_y.append(int(noiselabel))
                else:    
                    noise_y.append(int(original_y[i]))  
            self.y = noise_y
        
        self.atom_pool = atom_pool
        
        self.is_train = is_train

        print(f"Data Num: {len(self.y)}")

    def __len__(
        self
    ):
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        x_dummy = self.x.loc[i]

        if self.atom_pool != None:
            x_ = self.atom_pool.check_atoms(np.array(x_dummy))
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        y = self.y[i]
        
        x = torch.Tensor(x_dummy).float()
        return (x, x_), y

def load_data(
    dataset: str='yelp',
    data_dir: str='./data/',
    seed: int=7,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load data and split into train, valid, test dataset.
    """
    if dataset=='yelp':
        # Negative = 0, Positive = 1
        data_path = f'{data_dir}/yelp_review_polarity_csv'

        train_df = pd.read_csv(f'{data_path}/train.csv', header=None)
        train_df = train_df.rename(columns={0: 'label', 1: 'text'})
        train_df['label'] = train_df['label'] - 1
        _, train_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=seed,
            stratify=train_df['label']
        )

        test_df = pd.read_csv(f'{data_path}/test.csv', header=None)
        test_df = test_df.rename(columns={0: 'label', 1: 'text'})
        test_df['label'] = test_df['label'] - 1

        test_df = test_df.reset_index(drop=True)
        test_df, valid_df = train_test_split(
            test_df,
            test_size=0.5,
            random_state=seed,
            stratify=test_df['label']
        )

    elif dataset=='clickbait':
        # news = 0, clickbait = 1
        data_path = f'{data_dir}/clickbait_news_detection'

        train_df = pd.read_csv(f'{data_path}/train.csv')
        train_df = train_df.loc[
            train_df['label'].isin(['news', 'clickbait']),
            ['title', 'text', 'label']
        ]
        train_df = train_df.dropna()
        new_label = []
        for label in train_df['label']:
            if label == 'news':
                new_label.append(0)
            elif label == 'clickbait':
                new_label.append(1)
        train_df['label'] = new_label

        valid_df = pd.read_csv(f'{data_path}/valid.csv')
        valid_df = valid_df.loc[
            valid_df['label'].isin(['news', 'clickbait']),
            ['title', 'text', 'label']
        ]
        valid_df = valid_df.dropna()
        new_label = []
        for label in valid_df['label']:
            if label == 'news':
                new_label.append(0)
            elif label == 'clickbait':
                new_label.append(1)
        valid_df['label'] = new_label

        test_df, valid_df = train_test_split(
            valid_df,
            test_size=0.5,
            random_state=seed,
            stratify=valid_df['label']
        )

    elif dataset=='adult':
        # <=50K = 0, >50K = 1
        data_path = f'{data_dir}/adult'

        data_df = pd.read_csv(f'{data_path}/adult.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
        cat_map = get_tabular_category_map(data_df, dataset)

        number_data_df = numerize_tabular_data(data_df, cat_map, dataset)
        number_data_df = number_data_df[numerical_x_col + categorical_x_col + y_col]
        dummy_data_df = pd.get_dummies(number_data_df, columns=categorical_x_col)

        train_df, test_df = train_test_split(
            dummy_data_df,
            test_size=0.2,
            random_state=seed,
            stratify=number_data_df[y_col[0]]
        )

        valid_df, test_df = train_test_split(
            test_df,
            test_size=0.5,
            random_state=seed,
            stratify=test_df[y_col[0]]
        )
        
    elif dataset=='diabetes':
        # <=50K = 0, >50K = 1
        data_path = f'{data_dir}/diabetes'

        data_df = pd.read_csv(f'{data_path}/diabetes.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
        cat_map = get_tabular_category_map(data_df, dataset)

        number_data_df = numerize_tabular_data(data_df, cat_map, dataset)
        number_data_df = number_data_df[numerical_x_col + categorical_x_col + y_col]
        dummy_data_df = pd.get_dummies(number_data_df, columns=categorical_x_col)

        train_df, test_df = train_test_split(
            dummy_data_df,
            test_size=0.2,
            random_state=seed,
            stratify=number_data_df[y_col[0]]
        )

        valid_df, test_df = train_test_split(
            test_df,
            test_size=0.5,
            random_state=seed,
            stratify=test_df[y_col[0]]
        )
        
    elif dataset=='credit':
        # <=50K = 0, >50K = 1
        data_path = f'{data_dir}/credit'

        data_df = pd.read_csv(f'{data_path}/UCI_Credit_Card.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
        cat_map = get_tabular_category_map(data_df, dataset)

        number_data_df = numerize_tabular_data(data_df, cat_map, dataset)
        number_data_df = number_data_df[numerical_x_col + categorical_x_col + y_col]
        dummy_data_df = pd.get_dummies(number_data_df, columns=categorical_x_col)
        
        train_df, test_df = train_test_split(
            dummy_data_df,
            test_size=0.2,
            random_state=seed,
            stratify=number_data_df[y_col[0]]
        )

        valid_df, test_df = train_test_split(
            test_df,
            test_size=0.5,
            random_state=seed,
            stratify=test_df[y_col[0]]
        )
        
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    train_df, valid_df, test_df = [
        df.reset_index(
            drop=True
        ) for df in [train_df, valid_df, test_df]]

    return train_df, valid_df, test_df

def get_dataset_type(
    dataset: str='yelp'
) -> str:
    """
    Return the type of the dataset.
    """
    if dataset in NLP_DATASET:
        ret = 'nlp'
    elif dataset in TAB_DATASET:
        ret = 'tab'
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return ret

def get_base_type(
    base: str='bert'
) -> str:
    """
    Return the type of the base model.
    """
    if base in NLP_BASE:
        ret = 'nlp'
    elif base in TAB_BASE:
        ret = 'tab'
    else:
        raise ValueError(f'Base model {base} is not supported.')

    return ret

def get_label_column(
    dataset: str='yelp'
) -> str:
    """
    Return the label column of the dataset.
    """
    if dataset in ['yelp', 'clickbait']:
        label = 'label'
    elif dataset == 'adult':
        label = 'income'
    elif dataset == 'diabetes':
        label = 'Diabetes_binary'
    elif dataset == 'credit':
        label = 'default.payment.next.month'
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return label

def get_context_columns(
    dataset: str='yelp'
) -> List[str]:
    """
    Return the context columns of the dataset.
    This also can be used as position indicator.
    """
    if dataset == 'yelp':
        cols = ['text']
    elif dataset == 'clickbait':
        cols = ['title', 'text']
    elif dataset in ['adult', 'diabetes', 'credit']:
        categorical_x_col, numerical_x_col, _ = get_tabular_column_type(dataset)
        cols = categorical_x_col + numerical_x_col
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return cols

def get_class_names(
    dataset: str='yelp'
) -> List[str]:
    """
    Return the class names of the dataset.
    """
    if dataset == 'yelp':
        class_names = ['Negative', 'Positive']
    elif dataset == 'clickbait':
        class_names = ['news', 'clickbait']
    elif dataset == 'adult':
        class_names = ['<=50K', '>50K']
    elif dataset == 'diabetes':
        class_names = ['not diabetes', 'diabetes']
    elif dataset == 'credit':
        class_names = ['no', 'yes']
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    # For extensibility
    class_names = [str(c) for c in class_names]

    return class_names

def get_tabular_column_type(
    dataset: str='adult',
) -> List[str]:
    """
    Return the type of columns for the tabular dataset.
    """
    if dataset == 'adult':
        categorical_x_col = [
            'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
            'gender', 'native-country'
        ]

        numerical_x_col = [
            'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'
        ]

        y_col = ['income']
        
    elif dataset == 'diabetes':
        categorical_x_col = [
            'HighBP', 'HighChol', 'CholCheck',
            'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
            'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
            'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
        ]

        numerical_x_col = [
            'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 
        ]

        y_col = ['Diabetes_binary']
        
    elif dataset == 'credit':
        categorical_x_col = [
            'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0',
            'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
        ]

        numerical_x_col = [
            'LIMIT_BAL', 'AGE',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        ]

        y_col = ['default.payment.next.month']
        
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return categorical_x_col, numerical_x_col, y_col

def get_tabular_category_map(
    data_df: pd.DataFrame,
    dataset: str='adult',
) -> Dict[str, str]:
    """
    Return the category map that maps string-like categories to index
    """
    categorical_x_col, _, y_col = get_tabular_column_type(dataset)
    cat_map = {}
    
    for cat in categorical_x_col + y_col:
        cat_map[f'{cat}_idx2key'] = {}
        cat_map[f'{cat}_key2idx'] = {}
        count = Counter(data_df[cat])

        cat_keys = sorted(count.keys())
        for i, key in enumerate(cat_keys):
            cat_map[f'{cat}_idx2key'][i] = key
            cat_map[f'{cat}_key2idx'][key] = i

    return cat_map

def numerize_tabular_data(
    data_df: pd.DataFrame,
    cat_map: Dict[str, str],
    dataset: str='adult'
) -> pd.DataFrame:
    """
    Convert the given dataframe.
    Categorical column would become index from string,
    and numerical column would normalized by its maximum value.
    """
    def convert_key2idx(target, map_dict):
        return map_dict[target]

    categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)

    for cat in categorical_x_col + y_col:
        map_dict = cat_map[f'{cat}_key2idx']
        col = data_df[cat]
        new_col = col.apply(convert_key2idx, args=(map_dict, ))
        data_df[cat] = new_col

    data_df = data_df.astype(float)
    for col in numerical_x_col:
        data_df[col] = data_df[col] / max(data_df[col])

    return data_df

def get_tabular_numerical_threshold(
    data_df: pd.DataFrame,
    dataset: str='adult',
    interval: int=10
) -> Dict[str, list]:
    """
    Get thresholds to create atoms for each column of the tabular dataset.
    """
    _, numerical_x_col, _ = get_tabular_column_type(dataset)

    numerical_threshold = {}
    if dataset == 'adult':
        for col in numerical_x_col:
            numerical_threshold[col] = []
            if col in ['capital-gain', 'capital-loss']:
                target = data_df[col][data_df[col] != 0]
            else:
                target = data_df[col]

            target = target.to_numpy()
            for i in range(1, interval):
                percent = i * (100 / interval)
                numerical_threshold[col].append(np.percentile(target, percent))
    elif dataset in ['diabetes', 'credit']:
        for col in numerical_x_col:
            numerical_threshold[col] = []
            target = data_df[col]

            target = target.to_numpy()
            for i in range(1, interval):
                percent = i * (100 / interval)
                numerical_threshold[col].append(np.percentile(target, percent))
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return numerical_threshold

def get_tabular_numerical_max(
    data_df: pd.DataFrame,
    dataset: str='adult'
) -> Dict[str, float]:
    """
    Get the maximum value for each column of the tabular dataset.
    """
    _, numerical_x_col, _ = get_tabular_column_type(dataset)

    numerical_max = {}
    if dataset in ['adult', 'diabetes', 'credit']:
        for col in numerical_x_col:
            numerical_max[col] = data_df[col].describe()['max']
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return numerical_max

def load_tabular_info(
    dataset: str='adult',
    data_dir: str='./data/',
) -> Tuple[Dict[str, str], Dict[str, list], Dict[str, float]]:
    """
    Returns the data structures that contains information of the tabular dataset.
    """
    data_path = f'{data_dir}/{dataset}'

    if dataset=='adult':
        data_df = pd.read_csv(f'{data_path}/adult.csv')
    elif dataset=='diabetes':
        data_df = pd.read_csv(f'{data_path}/diabetes.csv')
    elif dataset=='credit':
        data_df = pd.read_csv(f'{data_path}/UCI_Credit_Card.csv')
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    cat_map = get_tabular_category_map(data_df, dataset)
    numerical_threshold = get_tabular_numerical_threshold(data_df, dataset=dataset)
    numerical_max = get_tabular_numerical_max(data_df, dataset=dataset)

    return cat_map, numerical_threshold, numerical_max

def create_dataset(
    data_df: pd.DataFrame,
    is_train: bool=True,
    dataset: str='yelp',
    atom_pool: object=None,
    atom_tokenizer: object=None,
    tf_tokenizer: object=None,
    config: object=None,
    noise_ratio: float=0.,
) -> object:
    """
    Create a dataset with the given dataframe.
    """
    if dataset == 'yelp':
        ret = YelpDataset(
            data_df,
            is_train=is_train,
            atom_pool=atom_pool,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            max_len=512,
            noise_ratio=noise_ratio,
        )
    elif dataset == 'clickbait':
        ret = ClickbaitDataset(
            data_df,
            is_train=is_train,
            atom_pool=atom_pool,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            max_len=512,
            noise_ratio=noise_ratio,
        )
    elif dataset == 'adult':
        ret = AdultDataset(
            data_df,
            is_train=is_train,
            atom_pool=atom_pool,
            noise_ratio=noise_ratio,
        )
    elif dataset == 'diabetes':
        ret = DiabetesDataset(
            data_df,
            is_train=is_train,
            atom_pool=atom_pool,
            noise_ratio=noise_ratio,
        )
    elif dataset == 'credit':
        ret = CreditDataset(
            data_df,
            is_train=is_train,
            atom_pool=atom_pool,
            noise_ratio=noise_ratio,
        )
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return ret

def create_dataloader(
    dataset: object,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> object:
    """
    Create a dataloader with the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=False,
        pin_memory=False,
    )

def get_single_context(
    target: pd.DataFrame,
    dataset: str,
    tabular_column_type: List[str],
    tabular_info: Tuple[dict, dict, dict],
):
    """
    Get a single context for given instance.
    Used to get an base contents for explanation of a single instance.
    """
    target_context = ''

    if dataset == 'yelp':
        target_context += f'text: {target["text"]}\n'
    elif dataset == 'clickbait':
        target_context += f'title: {target["title"]}\n'
        target_context += f'text: {target["text"]}\n'
    elif dataset in ['adult']:
        categorical_x_col, numerical_x_col, y_col = tabular_column_type
        cat_map, _, numerical_max = tabular_info
        
        for key in target.index:
            value = target[key]

            if key in numerical_x_col:
                target_context += f'{key}: {round(value * numerical_max[key], 1)}\n'
            elif key in y_col:
                continue
            else:
                if value == 1:
                    if dataset == 'adult':
                        context, category = key.split('_')
                        assert context in categorical_x_col
                        cur = cat_map[f'{context}_idx2key'][int(float(category))]
                        
                    target_context += f'{context}: {cur}\n'

    return target_context

def get_single_input(
    target: pd.DataFrame,
    dataset: str,
    atom_pool: object,
    tf_tokenizer: object=None,
    atom_tokenizer: object=None,
    max_len: int=512,
    additional_att: object=None,
) -> Tuple[torch.Tensor, ...]:
    """
    Get a single input for given instance.
    Used to get an explanation of a single instance.
    """
    assert atom_pool != None

    if dataset in NLP_DATASET:
        assert tf_tokenizer != None
        assert atom_tokenizer != None

        if dataset == 'yelp':
            text = target['text']
            text_pair = None

            text_bow = np.zeros(atom_tokenizer.vocab_size)
            text_count = Counter(atom_tokenizer.tokenize(target['text']))

            for word, count in dict(text_count).items():
                text_bow[word] = count
            bow = text_bow
        elif dataset == 'clickbait':
            text = target['title']
            text_pair = target['text']

            text_bow = np.zeros(atom_tokenizer.vocab_size)
            title_bow = np.zeros(atom_tokenizer.vocab_size)

            text_count = Counter(atom_tokenizer.tokenize(target['text']))
            title_count = Counter(atom_tokenizer.tokenize(target['title']))

            for word, count in dict(text_count).items():
                text_bow[word] = count

            for word, count in dict(title_count).items():
                title_bow[word] = count

            bow = np.concatenate((title_bow, text_bow))

        encoding = tf_tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = input_ids.squeeze(dim=0).long()
        attention_mask = attention_mask.squeeze(dim=0).long()

        x_ = atom_pool.check_atoms(bow)
        x_ = torch.Tensor(x_).long()

        ret = (input_ids, attention_mask, x_)

    elif dataset in TAB_DATASET:
        col_label = get_label_column(dataset)
        target = target.drop(index=[col_label])
        x = torch.Tensor(target).float()

        x_ = atom_pool.check_atoms(target)
        x_ = torch.Tensor(x_).long()

        ret = (x, x_)
    else:
        assert(0)

    return ret
