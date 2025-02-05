import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset,load_from_disk
from unit.config import config
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
import pandas as pd
class ContrastiveDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        if isinstance(idx, list):
            #print('idx is list')
            #print(idx)
            batch = [self._get_single_item(i) for i in idx]
            input_ids = torch.stack([item[0] for item in batch],dim=0)
            attention_mask = torch.stack([item[1] for item in batch],dim=0)
            labels = torch.tensor([item[2] for item in batch])

            #print('In ContrastiveDataset')
            #print(input_ids.size())
            #print(attention_mask.size())
            #print(labels.size())



            return torch.squeeze(input_ids), torch.squeeze(attention_mask), torch.squeeze(labels)
        else:
            #print('idx is single')
            return self._get_single_item(idx)
        
    def _get_single_item(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, label

class ContrastiveSampler(Sampler):
    def __init__(self, labels, embeddings, in_batch_size):
        self.labels = labels
        self.embeddings = embeddings
        self.in_batch_size = in_batch_size

    def __iter__(self):

        #batch = [15219, 24917, 420, 11797]
        #yield batch              
        while True:
            batch = []
            indices = list(range(len(self.labels)))
            random.shuffle(indices)
            for idx in indices:
                anchor_idx = idx
                anchor_label = self.labels[anchor_idx]
                anchor_embedding = self.embeddings[anchor_idx]

                pos_idx = self.get_positive_sample(anchor_idx, anchor_label, anchor_embedding)
                #print(f'pos_idx:{pos_idx}')
                neg_idx = self.get_negative_sample(anchor_idx, anchor_label, anchor_embedding)
                #print(f'neg_idx:{neg_idx}')
                hard_neg_idx = self.get_hard_negative_sample(anchor_idx, anchor_label, anchor_embedding)
                #print(f'hard_neg_idx:{hard_neg_idx}')

                if pos_idx is not None and neg_idx is not None and hard_neg_idx is not None:
                    batch.extend([anchor_idx, pos_idx, neg_idx, hard_neg_idx])                   
                    if len(batch) >= self.in_batch_size:
                        yield batch[:self.in_batch_size]
                        batch = batch[self.in_batch_size:]
                        # return batch size = 4
        

    def get_positive_sample(self, anchor_idx, anchor_label, anchor_embedding):

        if config.CONTRASTIVE_STRATEGY == 'mix':
            
            pos_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != anchor_idx]            
            pos_similarities = [1 - cosine(anchor_embedding.flatten(), self.embeddings[i].flatten()) for i in pos_indices]

            # mix
            if pos_similarities:
                top_pos_idx = pos_indices[np.argmax(pos_similarities)]
                return top_pos_idx
            else:
                pos_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != anchor_idx]
                pos_idx = random.choice(pos_indices)
                return pos_idx
            
        else:
            # random
            pos_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != anchor_idx]
            pos_idx = random.choice(pos_indices)
            return pos_idx

    def get_negative_sample(self, anchor_idx, anchor_label, anchor_embedding):

        if config.CONTRASTIVE_STRATEGY == 'mix':

            neg_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
            neg_similarities = [1 - cosine(anchor_embedding.flatten(), self.embeddings[i].flatten()) for i in neg_indices]

            # mix
            if neg_similarities:
                bottom_neg_idx = neg_indices[np.argmin(neg_similarities)]
                return bottom_neg_idx
        else:
            # random
            neg_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
            neg_idx = random.choice(neg_indices)
            return neg_idx

    def get_hard_negative_sample(self, anchor_idx, anchor_label, anchor_embedding):

        if config.CONTRASTIVE_STRATEGY == 'mix':

            hard_neg_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
            hard_neg_similarities = [1 - cosine(anchor_embedding.flatten(), self.embeddings[i].flatten()) for i in hard_neg_indices]

            # mix
            if hard_neg_similarities:
                top_hard_neg_idx = hard_neg_indices[np.argmax(hard_neg_similarities)]
                return top_hard_neg_idx
        else:
            # random
            hard_neg_indices = [i for i, label in enumerate(self.labels) if label != anchor_label and label != 0]
            hard_neg_idx = random.choice(hard_neg_indices)
            return hard_neg_idx
    
    def __len__(self):
        return len(self.labels) // self.in_batch_size

def get_embeddings(model, dataloader, device):
    model.to(device)
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader,desc='Conver to Embeddings'):
            input_ids, attention_mask, label = [t.to(device) for t in batch]

            if config.CONTRASTIVE_STRATEGY == 'random':
                embeddings.extend(input_ids.cpu().numpy())
            else:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                word_embeddings = output.last_hidden_state  # This contains the embeddings
                #print(word_embeddings)            
                embeddings.extend(word_embeddings.cpu().numpy())            
            labels.extend(label.cpu().numpy())
    return embeddings, labels


def clc_alpha_weight(labels):
    df = pd.DataFrame(labels,columns=['classes'])
    classes_count = df['classes'].value_counts().sort_index(ascending=True).tolist()
    total_num = len(df)
    init_weight = [0.25,0.75,0.75,0.75,0.75,0.75,0.75]
    frequency_rate = [x/total_num for x in classes_count]
    frequency_rate[0] = 1-frequency_rate[0]    
    print(frequency_rate)
    frequency_gap = [b-a for a,b in zip(frequency_rate,init_weight)]
    frequency_gap[0]=frequency_rate[0]
    weight = frequency_gap
    print(weight)    
    return weight

def get_dataloaders(batch_size, num_workers, max_length,dataset_path_or_name=None):
    # Get 1 data, sample to 4 batch data
    # dataset = load_dataset('imdb')

    # dataset = load_from_disk(f'{config.PROJECT_DIR}/data/{dataset_path_or_name}/Full')     

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)   


    if dataset_path_or_name == 'imdb':
        # 加载 IMDB 数据集    
        print(f'load dataset: {dataset_path_or_name}')
        
        dataset = load_from_disk(f'{config.PROJECT_DIR}/data/{dataset_path_or_name}/Full') 
        
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

    elif dataset_path_or_name.find('NLP4IF2019') != -1:
        # 加载数据集
        dataset_test_FD = load_from_disk(f'{config.PROJECT_DIR}/{config.TEST_DATASET}/dataset')

        if config.VERSION == 'ICK-L1':
            dataset_path = f'{config.PROJECT_DIR}/{dataset_path_or_name}/dataset'
            print(f'load dataset: {dataset_path}')
            dataset = load_from_disk(dataset_path)   
            train_texts = dataset['train_l1']['text']
            train_labels = dataset['train_l1']['classes']   

            weight = clc_alpha_weight(train_labels) 
            config.LOSS_ALPHA = weight          

            test_texts = dataset_test_FD['test']['text']            
            test_labels = dataset_test_FD['test']['classes']

        elif config.VERSION == 'ICK-L2':
            dataset_path = f'{config.PROJECT_DIR}/{dataset_path_or_name}/dataset'
            print(f'load dataset: {dataset_path}')
            dataset = load_from_disk(dataset_path) 
            train_texts = dataset['train_l2']['text']
            train_labels = dataset['train_l2']['classes']  

            weight = clc_alpha_weight(train_labels) 
            config.LOSS_ALPHA = weight

            test_texts = dataset_test_FD['test']['text']            
            test_labels = dataset_test_FD['test']['classes']

        elif config.VERSION == 'ICK-L3':
            dataset_path = f'{config.PROJECT_DIR}/{dataset_path_or_name}/dataset'
            print(f'load dataset: {dataset_path}')
            dataset = load_from_disk(dataset_path) 
            train_texts = dataset['train_l3']['text']
            train_labels = dataset['train_l3']['classes']

            weight = clc_alpha_weight(train_labels) 
            config.LOSS_ALPHA = weight

            test_texts = dataset_test_FD['test']['text']            
            test_labels = dataset_test_FD['test']['classes']
        
        elif config.VERSION == 'Full':
            dataset_train_FD = load_from_disk(f'{config.PROJECT_DIR}/{dataset_path_or_name}/Foundation(DF)/dataset')
            dataset_train_ICK = load_from_disk(f'{config.PROJECT_DIR}/{dataset_path_or_name}/Increasing(ICK)/Full/dataset')

            

            train_texts_FD = dataset_train_FD['train']['text']
            train_labels_FD = dataset_train_FD['train']['classes']

            train_texts_ICK = dataset_train_ICK['train']['text']
            train_labels_ICK = dataset_train_ICK['train']['classes']
            
            train_texts = np.concatenate((train_texts_FD,train_texts_ICK),axis=0)
            train_labels = np.concatenate((train_labels_FD,train_labels_ICK),axis=0)            

            weight = clc_alpha_weight(train_labels) 
            config.LOSS_ALPHA = weight

            test_texts = dataset_test_FD['test']['text']
            test_labels = dataset_test_FD['test']['classes']

        else:
            dataset_train_FD = load_from_disk(f'{config.PROJECT_DIR}/{dataset_path_or_name}/Foundation(DF)/dataset')

            train_texts = dataset_train_FD['train']['text']            
            train_labels = dataset_train_FD['train']['classes']

            weight = clc_alpha_weight(train_labels) 
            config.LOSS_ALPHA = weight

            test_texts = dataset_test_FD['test']['text']            
            test_labels = dataset_test_FD['test']['classes']

        
    else:
        print('Not Found Dataset Name')
        print(dataset_path_or_name)
        return False

    train_dataset = ContrastiveDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = ContrastiveDataset(test_texts, test_labels, tokenizer, max_length)

    # 预先获取训练数据的嵌入
    model = AutoModel.from_pretrained(config.MODEL_NAME)    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if config.CONTRASTIVE_STRATEGY == 'random':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    embeddings, labels = get_embeddings(model, train_loader, device)

    train_sampler = ContrastiveSampler(labels, embeddings, config.IN_BATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader 
