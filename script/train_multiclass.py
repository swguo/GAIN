import sys,os
sys.path.append('../')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from unit.config import config
from unit.dataset_multiclass import get_dataloaders
from unit.model import ContrastiveLearningModel
from unit.arguments import parse_args
from adapters import AutoAdapterModel

def train(args):
    config.PROJECT_NAME = args.project
    config.TASK_NAME = args.task_name
    config.VERSION = args.version
    config.MODEL_NAME = args.model_name_or_path
    config.CONTRASTIVE_STRATEGY = args.strategy
    load_checkpoint = args.load_checkpoint
    dataset_path_or_name = args.train_dataset
    config.TRAIN_DATASET = args.train_dataset
    config.TEST_DATASET = args.test_dataset

    wandb_logger = WandbLogger(project=config.PROJECT_NAME,name=f'{config.TASK_NAME}_{config.VERSION}_{config.CONTRASTIVE_STRATEGY}')    


    output_dirpath = os.path.join(config.PROJECT_DIR,   # /home/user_data
                                config.MODEL_BASEPATH,# models
                                config.PROJECT_NAME,  # IMDB
                                config.TASK_NAME,     # IMDB-multiclass
                                config.VERSION,       # v1
                                config.CONTRASTIVE_STRATEGY, # random or mix
                                config.MODEL_NAME     # bert-base-uncased
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dirpath,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    # EarlyStopping 回调，用于在 val_loss 连续 5 次上升时停止训练
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,  # 设置 patience 为 3
        mode='min',
        verbose=True
    )

    pretrain_model = AutoAdapterModel.from_pretrained(config.MODEL_NAME)

    train_loader, val_loader  = get_dataloaders(config.BATCH_SIZE,
                                                config.NUM_WORKERS, 
                                                config.MAX_LENGTH,
                                                dataset_path_or_name=dataset_path_or_name)
    
    if load_checkpoint != None :

        if config.VERSION == 'ICK-L1':
            base_version = 'FD'
        elif config.VERSION == 'ICK-L2':
            base_version = 'ICK-L1'
        elif config.VERSION == 'ICK-L3':
            base_version = 'ICK-L2'
        

        
        for param in pretrain_model.base_model.parameters():
            param.requires_grad = False     

        task_adapter = pretrain_model.load_adapter(os.path.join(config.PROJECT_DIR, 
                                                        'models',
                                                        config.PROJECT_NAME,
                                                        config.TASK_NAME,
                                                        base_version,  
                                                        config.CONTRASTIVE_STRATEGY, # random or mix                     
                                                        config.MODEL_NAME,
                                                        'task_adapter'
                                                        ), config="seq_bn")
        pretrain_model.set_active_adapters(task_adapter)

        
    else:

        
       

        '''
        # 配置 Bottleneck Adapters
        adapter_config = AdapterConfig(
            mh_adapter=True,  # Multi-Head Attention adapter
            output_adapter=True,
            reduction_factor=16,  # 調整 bottleneck 的大小
            non_linearity="relu",
        )'''

        for param in pretrain_model.base_model.parameters():
            param.requires_grad = False       


        # 添加並啟用適配器
        adapter_name = "task_adapter"
        pretrain_model.add_adapter(adapter_name, config="seq_bn")

        # 啟用適配器
        pretrain_model.set_active_adapters([adapter_name])

        # Add a matching classification head
        pretrain_model.add_classification_head(
            adapter_name,
            num_labels=config.NUM_CLASSES
        )

        pretrain_model.train_adapter(adapter_name)

    adapter_model = ContrastiveLearningModel(pretrain_model, 
                                    learning_rate=config.LEARNING_RATE, 
                                    weight_decay=config.WEIGHT_DECAY)
    print(adapter_model)
    
    devices = [0]

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="gpu",
        devices=devices,
        val_check_interval=config.VAL_CHECK_INTERVAL,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(adapter_model, train_loader,val_loader)

    adapter_model.save_adapter(output_dirpath, "task_adapter")

    print(pretrain_model)

    best_model_path = checkpoint_callback.best_model_path
    
    best_model = ContrastiveLearningModel.load_from_checkpoint(best_model_path,
                                                               adapter_model=pretrain_model)

    
    best_model.save_adapter(output_dirpath, 'task_adapter')

    return checkpoint_callback.best_model_path

if __name__ == "__main__":
    args = parse_args()
    train(args)
