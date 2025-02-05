from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from unit.datasets import TextDataset,MultiLabelTextDataset
from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder
from unit.config import config
import pandas as pd

def dataset_to_dataframe(dataset):
    data_list = []

    for i in range(len(dataset)):
        item = dataset[i]
        data_list.append(item)

    return pd.DataFrame(data_list)

def MultiLabelesDataMudule(tokenizer, max_length, dataset_path_or_name='jigsaw_toxicity'):
   
    print(f'load dataset: {dataset_path_or_name}')
    if dataset_path_or_name == 'jigsaw_toxicity':

        dataset = load_from_disk(f'{config.PROJECT_DIR}/data/jigsaw-toxic-comment-classification-challenge/Full')         
        columns_name = dataset['train'].column_names
        features_name = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        config.NUM_CLASSES = len(features_name)

        # training
        train_texts = dataset['train']['text']
        train_texts_df = pd.DataFrame([],columns=columns_name)
        for col in columns_name:
            train_texts_df[col] = dataset['train'][col]
        
        # print(train_texts_df.head(2))

        train_labels = train_texts_df[features_name].values.tolist()

        # validation
        test_texts = dataset['validation']['text']        
        test_texts_df = pd.DataFrame([],columns=columns_name)
        for col in columns_name:
            test_texts_df[col] = dataset['validation'][col]

        test_labels = test_texts_df[features_name].values.tolist()

    # 创建数据集实例  
    train_dataset = MultiLabelTextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = MultiLabelTextDataset(test_texts, test_labels, tokenizer, max_length)
    test_dataset = MultiLabelTextDataset(test_texts, test_labels, tokenizer, max_length)

    # 创建 DataLoader
    num_workers = 19  # 设置 num_workers 参数以提高性能   
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def MultiClassDataMudule(tokenizer, max_length, dataset_path_or_name='imdb'):
    
    if dataset_path_or_name == 'imdb':
        # 加载 IMDB 数据集    
        print(f'load dataset: {dataset_path_or_name}')
        
        dataset = load_from_disk(f'{config.PROJECT_DIR}/data/{dataset_path_or_name}/Full')     
        
        # 标签编码
        label_encoder = LabelEncoder()
        train_texts = dataset['train']['text']
        train_labels = label_encoder.fit_transform(dataset['train']['label'])
        test_texts = dataset['test']['text']
        test_labels = label_encoder.transform(dataset['test']['label'])

    elif dataset_path_or_name.find('NLP4IF2019') != -1:
        # 加载数据集    
        print(f'load dataset: {dataset_path_or_name}')
        
        dataset = load_from_disk(f'{config.PROJECT_DIR}/data/{dataset_path_or_name}/dataset')     
        
        # 标签编码
        label_encoder = LabelEncoder()
        train_texts = dataset['train']['text']
        train_labels = label_encoder.fit_transform(dataset['train']['classes'])

        if config.VERSION == 'ICK-L1':
            print('test_l1')
            test_texts = dataset['test_l1']['text']
            test_labels = label_encoder.transform(dataset['test_l1']['classes'])
        elif config.VERSION == 'ICK-L2':
            print('test_l2')
            test_texts = dataset['test_l2']['text']
            test_labels = label_encoder.transform(dataset['test_l2']['classes'])
        elif config.VERSION == 'ICK-L3':
            print('test_l3')
            test_texts = dataset['test_l3']['text']
            test_labels = label_encoder.transform(dataset['test_l3']['classes'])
        else:            
            print('Not Found Dataset Version')
            print(config.VERSION)
            return False

    else:
        print('Not Found Dataset Name')
        print(dataset_path_or_name)
        return False
    
    # 创建数据集实例  
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)

    # 创建 DataLoader
    num_workers = 19  # 设置 num_workers 参数以提高性能   
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=num_workers)

    return train_loader, val_loader, test_loader