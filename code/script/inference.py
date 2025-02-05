
import sys
sys.path.append('../')
from unit.config import config
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm
import wandb
from unit.model import ContrastiveLearningModel
from unit.arguments import infr_parse_args
import torch
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from unit.datasets import TextDataset, MultiLabelTextDataset
from transformers import AutoTokenizer
import torch.nn.functional as F
from adapters import AutoAdapterModel
# 預測產生結果、多類別 Multi-Labels
def MultiLabelsPredit(model, dataloader, tokenizer, device):
    model.to(device)
    model.eval()
    probas = []
    labels = []
    texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, label = [t.to(device) for t in batch]
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prob = torch.sigmoid(output.squeeze()).cpu()            
            probas.extend(prob.cpu().numpy())
            labels.extend(label.cpu().numpy())
            #print(input_ids)
            text = tokenizer.batch_decode(input_ids.cpu(),skip_special_tokens=True)
            #print(text)
            texts.extend(text)

    return np.array(texts), np.array(probas), np.array(labels)

# 預測產生結果、多類別 Multi-Class
def MultiClassPredit(model, dataloader, tokenizer, device):
    model.to(device)
    model.eval()
    y_probas = []
    preds = []
    labels = []
    texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, label = [t.to(device) for t in batch]
            # print(input_ids, attention_mask, label)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            pred = torch.argmax(output.logits, dim=1)
            y_proba = F.softmax(output.logits, dim=-1)

            preds.extend(pred.round().cpu().numpy())            
            y_probas.extend(y_proba.cpu().numpy())
            labels.extend(label.cpu().numpy())

            #print(input_ids)
            text = tokenizer.batch_decode(input_ids.cpu(),skip_special_tokens=True)
            #print(text)
            texts.extend(text)

    return np.array(texts), np.array(y_probas), np.array(preds), np.array(labels)

def save(base_model_name,test_texts,y_true,y_pred,test_probs,file_name='prediction.csv'):
    output_dir = os.path.join(config.PROJECT_DIR,
                                'models',
                                config.PROJECT_NAME,
                                config.TASK_NAME,
                                config.VERSION,
                                config.CONTRASTIVE_STRATEGY, # random or mix                            
                                base_model_name,
                                'results')

    # 保存结果到模型路径中的 results 目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    prediction_df = pd.DataFrame([],columns=['text','y_pred','y_true','y_proba'])
    
    prediction_df['text'] = test_texts.tolist()    
    prediction_df['y_pred'] = y_pred
    prediction_df['y_true'] = y_true    
    prediction_df['y_proba'] = test_probs.tolist()

    
    prediction_df.to_csv(f'{output_dir}/{file_name}',index=None) 

def main(args):
    config.PROJECT_NAME = args.project
    config.TASK_NAME = args.task_name
    config.VERSION =  args.version
    base_model_name = args.base_model_name
    best_checkpoint = args.best_checkpoint
    dataset_path_or_name = args.dataset_path_or_name
    config.CONTRASTIVE_STRATEGY = args.strategy

    tokenizer = AutoTokenizer.from_pretrained(base_model_name) # 加载预训练的模型和 tokenizer    

    '''
    # 加载最佳模型检查点
    best_model = ContrastiveLearningModel.load_from_checkpoint(
                    os.path.join(config.PROJECT_DIR, 
                                'models',
                                config.PROJECT_NAME,
                                config.TASK_NAME,
                                config.VERSION,  
                                config.CONTRASTIVE_STRATEGY, # random or mix                     
                                base_model_name,
                                best_checkpoint,)
    )
    '''
    best_model = AutoAdapterModel.from_pretrained(base_model_name)
    task_adapter = best_model.load_adapter(os.path.join(config.PROJECT_DIR, 
                                            'models',
                                            config.PROJECT_NAME,
                                            config.TASK_NAME,
                                            config.VERSION,  
                                            config.CONTRASTIVE_STRATEGY, # random or mix                     
                                            base_model_name,
                                            'task_adapter'
                                            ), config="seq_bn")
    best_model.set_active_adapters(task_adapter)
    
    print(best_model)

    # 进行推理和评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    dataset = load_from_disk(f'../../data/{dataset_path_or_name}') 

    columns_name = dataset['test'].column_names

    print('Dataset Information')
    print(dataset)

    if  config.PROJECT_NAME == 'Jigsaw':
        features_name = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        print('Features Name')
        print(features_name)

        test_texts = dataset['validation']['text']        
        test_texts_df = pd.DataFrame([],columns=columns_name)
        for col in columns_name:
            test_texts_df[col] = dataset['validation'][col]

        test_labels = test_texts_df[features_name].values.tolist()

        # 创建数据集实例
        test_dataset = MultiLabelTextDataset(test_texts, test_labels, tokenizer)

        # 创建 DataLoader
        num_workers = 19  # 设置 num_workers 参数以提高性能
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers)

        test_texts, test_probs, test_labels  = MultiLabelsPredit(best_model, test_loader, tokenizer, device) 

        y_preds = np.zeros(test_probs.shape)
        print(f'result of predit shape {y_preds.shape}') 
        micro_thresholds = 0.5    
        y_preds[np.where(test_probs >= micro_thresholds)] = 1

        y_pred = y_preds.astype(int).tolist()
        y_true = test_labels.astype(int).tolist()

    elif config.PROJECT_NAME == 'IMDB':       

        # 标签编码
        label_encoder = LabelEncoder()
        train_texts = dataset['train']['text']
        train_labels = label_encoder.fit_transform(dataset['train']['label'])
        test_texts = dataset['test']['text']
        test_labels = label_encoder.transform(dataset['test']['label'])

        # 创建数据集实例        
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # 创建 DataLoader
        num_workers = 19  # 设置 num_workers 参数以提高性能
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers)

        # np.array(texts), np.array(y_probas), np.array(preds), np.array(labels)
        test_texts, test_probs, test_preds, test_labels  = MultiClassPredit(best_model, test_loader, tokenizer, device) 
        
        y_pred = test_preds.astype(int).tolist()
        y_true = test_labels.tolist()

        save(base_model_name,test_texts,y_true,y_pred,test_probs,file_name='prediction.csv')

    elif config.PROJECT_NAME == 'PersuTech-adapter-4c-DIL':


        # 标签编码
        label_encoder = LabelEncoder()         
        test_texts_FD = dataset['test']['text']
        test_labels_FD = label_encoder.fit_transform(dataset['test']['classes'])
    
        # 创建数据集实例        
        test_dataset_FD = TextDataset(test_texts_FD, test_labels_FD, tokenizer)

        # 创建 DataLoader
        num_workers = 19  # 设置 num_workers 参数以提高性能
        test_loader_FD = DataLoader(test_dataset_FD, batch_size=4, num_workers=num_workers)

        # np.array(texts), np.array(y_probas), np.array(preds), np.array(labels)
        test_texts_FD, test_probs_FD, test_preds_FD, test_labels_FD  = MultiClassPredit(best_model, test_loader_FD, tokenizer, device) 
        
        y_pred_FD = test_preds_FD.astype(int).tolist()
        y_true_FD = test_labels_FD.tolist()

        save(base_model_name,test_texts_FD,y_true_FD,y_pred_FD,test_probs_FD,file_name=f'prediction_{config.VERSION}.csv')

    else:
        print(f'No {config.PROJECT_NAME} project')
        return False
    
       

if __name__ == '__main__':
    args = infr_parse_args()
    main(args)