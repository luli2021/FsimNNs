import os
import torch
from torchmetrics.classification import Accuracy, Precision, Recall

from torch_geometric.loader import DataLoader
from datasets.customized_dataset import CustomGraphDataset, DisplayDatasetInfo
from train import GNNLightning
from models.models import get_model

from utils.utils import get_logger
from utils.config import load_config

import pandas as pd


logger = get_logger('FsimNNPrediction')


def evaluate_model(model, test_loader, output_path):
    
    model.eval()  # Set model to evaluation mode
    
    results = []
    all_preds =[]
    all_labels = []

    accuracy = Accuracy(task="multiclass", num_classes=2) 
    precision = Precision(task="multiclass", num_classes=2, average="macro")
    recall = Recall(task="multiclass", num_classes=2, average="macro")

    with torch.no_grad():  
        for data in test_loader:
            data = data.to(model.device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            if data.test_mask.sum() > 0:
                preds = preds[data.test_mask].cpu().numpy()
                y = data.y[data.test_mask].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y)
               
                node_ids = data.node_ids[data.test_mask].cpu().numpy()
                for i, node_id in enumerate(node_ids):
                    results.append({
                        "source_path": data.source_path[0],
                        "graph_id": data.graph_id[0],
                        "node_id": node_id,
                        "predicted_label": preds[i],
                        "true_label": y[i]
                    })



    # Save results to a CSV file
    output_file = os.path.join(output_path,'results.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    
    # Convert lists to tensors before computing metrics
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    # Compute metrics
    acc = accuracy(all_preds, all_labels)
    prec = precision(all_preds, all_labels)
    rec = recall(all_preds, all_labels)
    
    metrics_df = pd.DataFrame({
        "Accuracy": [acc.item()],
        "Precision": [prec.item()],
        "Recall": [rec.item()]
    })
        
    # Save metrics to a CSV file
    scores_file = os.path.join(output_path,'scores.csv')
    metrics_df.to_csv(scores_file, index=False)
    logger.info(f"Metrics saved to {scores_file}")    

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")

    return acc, prec, rec




def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)   

    dataset_config = config["dataset"]
    root_dir = dataset_config["root_dir"]
    train_val_dirs = dataset_config["train_val_dirs"]
    test_dirs = dataset_config["test_dirs"]
    split_mode = dataset_config.get("split_mode", "spatio")

    model_config = config["model"]
    edge_vector_size = model_config.get("edge_vector_size", 10)
    time_window_size = model_config.get("time_window_size", 60)    
    
    test_dataset = CustomGraphDataset(root_dir, train_val_dirs, test_dirs, split_mode=split_mode, phase="test", max_node_feat_len=time_window_size, max_edge_feat_len=edge_vector_size)
    DisplayDatasetInfo(test_dataset, "test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)    
    
    
    best_model_path = os.path.join(config["model_path"], "best_model.ckpt") 
    logger.info(f"Best model is read from: {best_model_path}")
    
    model = get_model(config)
    model = GNNLightning.load_from_checkpoint(best_model_path, model=model)        
                      
    output_path = config["output_path"]
    evaluate_model(model, test_loader, output_path)




if __name__ == '__main__':
    main()
