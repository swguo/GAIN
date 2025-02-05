import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from config import config
import numpy as np

class ContrastiveDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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
        label = torch.tensor(label, dtype=torch.float)
        return input_ids, attention_mask, label

class ContrastiveSampler(Sampler):
    def __init__(self, labels, embeddings, batch_size):
        self.labels = labels
        self.embeddings = embeddings
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = []
            indices = list(range(len(self.labels)))
            random.shuffle(indices)
            for idx in indices:
                anchor_idx = idx
                anchor_label = self.labels[anchor_idx]
                anchor_embedding = self.embeddings[anchor_idx]

                pos_idx = self.get_positive_sample(anchor_idx, anchor_label, anchor_embedding)
                neg_idx = self.get_negative_sample(anchor_idx, anchor_label, anchor_embedding)
                hard_neg_idx = self.get_hard_negative_sample(anchor_idx, anchor_label, anchor_embedding)

                if pos_idx is not None and neg_idx is not None and hard_neg_idx is not None:
                    batch.extend([anchor_idx, pos_idx, neg_idx, hard_neg_idx])
                    if len(batch) >= self.batch_size:
                        yield batch[:self.batch_size]
                        batch = batch[self.batch_size:]

    def get_positive_sample(self, anchor_idx, anchor_label, anchor_embedding):
        pos_indices = [i for i, label in enumerate(self.labels) if (label == anchor_label).all() and i != anchor_idx]
        for pos_idx in pos_indices:
            pos_embedding = self.embeddings[pos_idx]
            cos_sim = np.dot(anchor_embedding, pos_embedding) / (np.linalg.norm(anchor_embedding) * np.linalg.norm(pos_embedding))
            if cos_sim > 0.8:
                return pos_idx
        return None

    def get_negative_sample(self, anchor_idx, anchor_label, anchor_embedding):
        neg_indices = [i for i, label in enumerate(self.labels) if not (label == anchor_label).all()]
        for neg_idx in neg_indices:
            neg_embedding = self.embeddings[neg_idx]
            cos_sim = np.dot(anchor_embedding, neg_embedding) / (np.linalg.norm(anchor_embedding) * np.linalg.norm(neg_embedding))
            if cos_sim < 0.5:
                return neg_idx
        return None

    def get_hard_negative_sample(self, anchor_idx, anchor_label, anchor_embedding):
        hard_neg_indices = [i for i, label in enumerate(self.labels) if not (label == anchor_label).all()]
        for hard_neg_idx in hard_neg_indices:
            hard_neg_embedding = self.embeddings[hard_neg_idx]
            cos_sim = np.dot(anchor_embedding, hard_neg_embedding) / (np.linalg.norm(anchor_embedding) * np.linalg.norm(hard_neg_embedding))
            if cos_sim > 0.8:
                return hard_neg_idx
        return None

    def __len__(self):
        return len(self.labels) // self.batch_size

def get_embeddings(model, dataloader, device):
    model.to(device)
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, label = [t.to(device) for t in batch]
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.extend(output.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return embeddings, labels

def get_dataloaders(batch_size, num_workers, max_length):
    dataset = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # 将多类标签转换为多标签格式
    num_classes = max(train_labels) + 1
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    train_dataset = ContrastiveDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = ContrastiveDataset(test_texts, test_labels, tokenizer, max_length)

    # 预先获取训练数据的嵌入
    model = AutoModel.from_pretrained(config.MODEL_NAME)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings, labels = get_embeddings(model, train_loader, device)

    train_sampler = ContrastiveSampler(labels, embeddings, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader 
