import torch
from sklearn.metrics import roc_auc_score
import json
import wandb
from unit.config import config
from unit.dataset_multilabel import get_dataloaders
from unit.model import ContrastiveLearningModel

def inference(model, dataloader, device):
    model.to(device)
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, label = [t.to(device) for t in batch]
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.extend(embedding.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return embeddings, labels

def evaluate(best_model_path):
    _, test_loader = get_dataloaders(config.BATCH_SIZE, config.NUM_WORKERS, config.MAX_LENGTH)
    best_model = ContrastiveLearningModel.load_from_checkpoint(best_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings, labels = inference(best_model, test_loader, device)
    
    # 在这里添加您想要的评估方法，例如计算 ROC AUC 分数
    auc_score = roc_auc_score(labels, embeddings)
    
    results = {
        'auc_score': auc_score
    }

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    with open(os.path.join(config.RESULTS_DIR, 'inference_results.json'), 'w') as f:
        json.dump(results, f)

    wandb.log({
        "auc_score": auc_score
    })

if __name__ == "__main__":
    best_model_path = "checkpoints/best-checkpoint.ckpt"
    evaluate(best_model_path)
