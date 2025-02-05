import torch
from transformers import AutoModelForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AdamW
import pytorch_lightning as pl
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_auc_score, classification_report
import json
import numpy as np
import pandas as pd
import os,wandb
import json
from tqdm.auto import tqdm
from unit.config import config
import torch.nn.functional as F
from unit.helper import Eval
from unit.loss import ResampleLoss,focal_loss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from torch import nn
import pytorch_lightning as pl
from unit.config import config
from adapters  import AutoAdapterModel, AdapterConfig

class ContrastiveLearningModel(pl.LightningModule):
    def __init__(self, adapter_model, dropout_prob=0.3, learning_rate=2e-5, weight_decay=0.01):
        super(ContrastiveLearningModel, self).__init__()
        self.save_hyperparameters()
        
        self.encoder = adapter_model     

        print('focal loss, alpha')
        print(config.LOSS_ALPHA)  

        self.val_preds = []
        self.val_labels = []

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(input_ids=input_ids.squeeze(), 
                               attention_mask=attention_mask.squeeze())
        
        # print(outputs)
        cls_output = outputs.logits  # 获取 [CLS] 标记的输出
        
        return cls_output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        
        outputs = self(input_ids.squeeze(), attention_mask.squeeze())
        #loss = self.contrastive_loss(outputs, labels)
        '''
        print('in train')
        print('outputs') # [4,2]
        print(outputs)
        print('labels') # [4]
        print(labels)
        '''
        loss = torch.nn.functional.cross_entropy(outputs,labels.squeeze())
        #loss = focal_loss(alpha=config.LOSS_ALPHA,
        #                  gamma=config.LOSS_GAMMA,
        #                  num_classes=config.NUM_CLASSES)(outputs,labels.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)

        pred = torch.argmax(outputs, dim=1)

        self.val_preds.append(pred.cpu())
        self.val_labels.append(labels.cpu())


        #loss = self.contrastive_loss(outputs, labels)
        loss = torch.nn.functional.cross_entropy(outputs,labels)
        #loss = focal_loss(alpha=config.LOSS_ALPHA,
        #                  gamma=config.LOSS_GAMMA,
        #                  num_classes=config.NUM_CLASSES)(outputs,labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        return loss
   
    def on_validation_epoch_end(self):
        val_preds = torch.cat(self.val_preds).numpy()
        val_labels = torch.cat(self.val_labels).numpy()

        #print('val_preds')
        #print(val_preds.size())
        #print(val_preds)
        #print('val_labels')
        #print(val_labels.size())
        #print(val_labels)

        precision = precision_score(val_labels, val_preds, average='micro')
        recall = recall_score(val_labels, val_preds, average='micro')
        f1 = f1_score(val_labels, val_preds, average='micro')
        accuracy = accuracy_score(val_labels, val_preds)

        self.log('val_precision', precision, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_recall', recall, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_f1', f1, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_accuracy', accuracy, prog_bar=True, logger=True,sync_dist=True)
        
        self.val_preds.clear()
        self.val_labels.clear()
  
    def contrastive_loss(self, outputs, labels):
        anchor, positive, negative, hard_negative = outputs[0::4], outputs[1::4], outputs[2::4], outputs[3::4]
        cos_sim = nn.CosineSimilarity(dim=-1)
        pos_sim = cos_sim(anchor, positive)
        neg_sim = cos_sim(anchor, negative)
        hard_neg_sim = cos_sim(anchor, hard_negative)
        margin = 1.0
        loss = torch.mean(torch.relu(margin - pos_sim + neg_sim) + torch.relu(margin - pos_sim + hard_neg_sim))
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def save_adapter(self,output_dirpath, task_adapter):

        self.encoder.save_adapter(f'{output_dirpath}/{task_adapter}', task_adapter)