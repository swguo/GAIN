import os

class Config:
    # 全局配置
    MODEL_NAME = 'cardiffnlp/twitter-roberta-large-2022-154m' #'FacebookAI/roberta-large'
    TRAIN_DATASET = ''
    TEST_DATASET = ''
    MAX_LENGTH = 256
    BATCH_SIZE = 1
    IN_BATCH_SIZE = 4
    NUM_WORKERS = 9
    NUM_CLASSES = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 10
    VAL_CHECK_INTERVAL = 10  # 每500步进行一次验证
    PROJECT_NAME = "PersuTech-adapter-4c-DIL"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    MODEL_BASEPATH = "models"
    TASK_NAME = "Naive"
    VERSION = "v1"
    PROJECT_NAME_MULTICLASS = "PersuTech-4c-DIL"
    PROJECT_NAME_MULTILABEL = "Jigsaw-contra"
    PROJECT_DIR = '/user_data/workspace/CL-PersuTech-4c-DIL/1_adapter'
    FEATURES_NAME = ['None','Ethos','Logos','Pathos','classes']
    CONTRASTIVE_STRATEGY = 'random' # random, topk, mix
    LOSS_GAMMA = 2
    LOSS_ALPHA = [0.4,0.4,0.6,0.4]
    # 确保目录存在
    #os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    #os.makedirs(RESULTS_DIR, exist_ok=True)

config = Config()
