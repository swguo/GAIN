import sys,os
sys.path.append('../')
import torch
from sklearn.metrics import roc_auc_score
import json,os
import wandb
from unit.config import config
from unit.dataset_multiclass import get_dataloaders
from unit.model import ContrastiveLearningModel
import pandas as pd
from unit.arguments import eval_parse_args
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_confusion_matrix(confusion_matrix,output_dir,task):
  
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  
  plt.title(f'{task}', fontsize = 18) 
  plt.xlabel('Predict', fontsize = 18)
  plt.ylabel('Ground Truth', fontsize = 18) 
  task = task.replace("/","-")
  task = task.replace(' ','')
  hmap.figure.savefig(f"{output_dir}/{task}.png")
  plt.clf()

def evaluate(y_true,y_pred,y_proba):

    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true,y_proba, multi_class='ovr')

    return precision, recall, f1, accuracy, roc

if __name__ == "__main__":
    args = eval_parse_args()
    file_name = args.dataset_path_or_name
    config.PROJECT_NAME = args.project
    config.TASK_NAME = args.task_name
    dataset_path_or_name = args.dataset_path_or_name

    
    config.VERSION =  args.version 

    base_model_name = args.base_model_name
    
    config.CONTRASTIVE_STRATEGY = args.strategy
    
    model_path = os.path.join(config.PROJECT_DIR,
                            'models',
                            config.PROJECT_NAME,
                            config.TASK_NAME,
                            config.VERSION, 
                            config.CONTRASTIVE_STRATEGY, # random or mix                          
                            base_model_name,
                            )
    eval_file_path = os.path.join(model_path,'results',dataset_path_or_name)
    output_dir = os.path.join(model_path,'results')
    
    df = pd.read_csv(eval_file_path,converters={"y_proba": literal_eval})
    
    y_true = df['y_true'].to_list()
    y_pred = df['y_pred'].to_list()
    y_proba = df['y_proba'].to_list()
    # print(evaluate(y_true,y_pred,y_proba))
    class_report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    class_rep_df = pd.DataFrame(class_report).transpose()   

    target_name = ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6']
    labels = [0,1,2,3,4,5,6]
    confusion_matrix = confusion_matrix(y_true, y_pred, labels = labels)


    class_rep_df.to_csv(f'{output_dir}/class_report-{config.VERSION}.csv') 
    show_confusion_matrix(confusion_matrix,output_dir,task=f'{config.PROJECT_NAME}-{config.VERSION}')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 找到唯一的类别
    classes = np.unique(y_true)

    # 创建一个字典来存储每个类别的准确率
    class_accuracy = {}

    for cls in classes:
        # 找到属于当前类别的索引
        indices = np.where(y_true == cls)
        
        # 计算当前类别的准确率
        accuracy = accuracy_score(y_true[indices], y_pred[indices])
        
        # 存储结果
        class_accuracy[cls] = accuracy

    # 打印每个类别的准确率
    for cls, acc in class_accuracy.items():
        print(f"Class {cls} Accuracy: {acc:.4f}")