# sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import evaluate
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,auc

class Eval():

    def __init__(self,mode='multilabels'):   
        self.mode = mode
        if mode == 'multilabels':
            print(f'Eval mode is {mode}')
            self.eval_f1 = evaluate.load("f1","multilabel")
            self.eval_acc = evaluate.load("accuracy","multilabel")
            self.eval_p = evaluate.load("precision","multilabel")
            self.eval_r = evaluate.load("recall","multilabel")
        else:      
            print(f'Eval mode is {mode}')      
            self.eval_f1 = evaluate.load("f1")
            self.eval_acc = evaluate.load("accuracy")
            self.eval_p = evaluate.load("precision")
            self.eval_r = evaluate.load("recall")

    def classifytaskEval(self,y_true,y_pred,y_proba=None): 
              
        micro_p = self.eval_p.compute(predictions=y_pred, 
                                        references=y_true,
                                        average="micro")['precision']
        
        weighted_p = self.eval_p.compute(predictions=y_pred, 
                                           references=y_true,
                                           average="weighted")['precision']

        micro_r = self.eval_r.compute(predictions=y_pred, 
                                        references=y_true,
                                        average="micro")['recall']
        
        weighted_r = self.eval_r.compute(predictions=y_pred, 
                                           references=y_true,
                                           average="weighted")['recall']       
        
        f1_metric = self.eval_f1.compute(predictions=y_pred, 
                                    references=y_true,
                                    average="micro")["f1"]  
        
        acc_metric = self.eval_acc.compute(predictions=y_pred, 
                                     references=y_true)["accuracy"]

        results_metric = {"precision(micro)": micro_p, "precision(weighted)": weighted_p, 
                          "recall(micro)": micro_r, "recall(weighted)": weighted_r, 
                          "f1":f1_metric,
                          "accuracy": acc_metric}
        '''
        if y_proba.all() != None:            
            pos_label = len(np.unique(y_true))
            if self.mode == 'multilabels':
                print(f'roc_auc_score mode is {self.mode}')
                roc_auc = roc_auc_score(y_true, y_proba.tolist(), average='micro')
            else:
                print(f'roc_auc_score mode is {self.mode}')
                roc_auc = roc_auc_score(y_true, y_proba.tolist(), multi_class='ovr')
            #fpr, tpr, thresholds = roc_curve(y, scores, pos_label=pos_label)
            #roc_auc = auc(fpr, tpr)
            results_metric['auc'] = roc_auc
        '''

        return results_metric
