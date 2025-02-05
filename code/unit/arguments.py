import os,argparse

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='bert-base-cased',
        choices=[   'bert-base-cased',
                    'bert-base-uncased',
                    "bert-large-uncased",
                    "vinai/bertweet-base",
                    "vinai/bertweet-large",
                    "xlm-roberta-base",
                    "xlm-roberta-large",
                    "sdadas/xlm-roberta-large-twitter",
                    "Twitter/twhin-bert-large",
                    'cardiffnlp/twitter-roberta-large-2022-154m',
                    'FacebookAI/roberta-large'
        ],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="load pre-train model form local disk",
        required=False,
    ) 
    parser.add_argument(
        "--output_dir",
        type=str,
        default='models',
        help='model path',
        required=True
    )
    parser.add_argument(
        '--project',
        type=str,
        default='classification task',
        required=True
    )
    parser.add_argument(
        '--train_dataset',
        type=str,
        help='pytorch dataset',
        required=True        
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        help='pytorch dataset',
        required=True        
    )
    parser.add_argument(
        '--task_name',
        type=str,
        help='ex : multi-label-global',
        required=True        
    )
    parser.add_argument(
        '--version',
        type=str,
        help='v1',
        required=True
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='random or mix',
        required=True
    )
    args = parser.parse_args()

    return args

def infr_parse_args():
    
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default='bert-base-cased',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--best_checkpoint",
        type=str,
        default='../models/IMDB/bert-base-cased/v1',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )    
    parser.add_argument(
        '--project',
        type=str,
        default='classification task',
        required=True
    )
    parser.add_argument(
        '--dataset_path_or_name',
        type=str,
        help='pytorch dataset',
        required=True        
    )
    parser.add_argument(
        '--task_name',
        type=str,
        help='ex : multi-label-global',
        required=True        
    )
    parser.add_argument(
        '--version',
        type=str,
        help='v1',
        required=True
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='random or mix',
        required=True
    )

    args = parser.parse_args()

    return args

def eval_parse_args():
    
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default='bert-base-cased',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        '--project',
        type=str,
        default='classification task',
        required=True
    )
    parser.add_argument(
        '--dataset_path_or_name',
        type=str,
        help='pytorch dataset',
        required=True        
    )
    parser.add_argument(
        '--task_name',
        type=str,
        help='ex : multi-label-global',
        required=True        
    )
    parser.add_argument(
        '--version',
        type=str,
        help='v1',
        required=True
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='random or mix',
        required=True
    )
    args = parser.parse_args()

    return args