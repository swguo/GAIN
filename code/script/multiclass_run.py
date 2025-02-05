import sys
sys.path.append('../')
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModel, AutoTokenizer, AdamW
from unit.model import MultiClassEncoderWithLinearLayer
from script.train import do_train
from unit.data_mudule import MultiClassDataMudule
from unit.arguments import parse_args
from unit.config import config


def main(args):

    project = args.project #"IMDB"
    max_length = config.MAX_LENGTH

    config.TASK_NAME = args.task_name
    config.VERSION = args.version

    wandb_logger = WandbLogger(project=project,name=f'{config.TASK_NAME}_{config.VERSION}')    
    model_name = args.model_name_or_path #"bert-base-uncased"  # 可更改为任意 Hugging Face 支持的模型
    tokenizer = AutoTokenizer.from_pretrained(model_name) # 加载预训练的模型和 tokenizer
    dataset_path_or_name = args.train_dataset #'imdb'  
    train_loader, val_loader, test_loader = MultiClassDataMudule(tokenizer, max_length,dataset_path_or_name)

    
    config.PROJECT_NAME = project
    config.MODEL_NAME = model_name

    # 创建模型实例
    num_classes =  config.NUM_CLASSES  # IMDB 数据集为二分类任务
    model = MultiClassEncoderWithLinearLayer(model_name, num_classes)

    #try:
    checkpoint_callback, trainer = do_train( 
                                    model,
                                    train_loader,
                                    val_loader,
                                    wandb_logger
                                    )
                            
    # 加载最佳模型检查点并进行评估
    #best_model_path = checkpoint_callback.best_model_path
    #trainer.test(ckpt_path=best_model_path, dataloaders=test_loader)

    return True
    #except Exception as error:
        # handle the exception
    #    print("An exception occurred:", error) # An exception occurred: division by zero

    #    return False

if __name__ == '__main__':
    args = parse_args()
    run_status = main(args)
    if run_status:
        print('training sucesses')
    else:
        print('training error')
    sys.exit()
