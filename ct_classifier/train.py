'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
    2025 Katie Grabowski
'''

import os
import argparse
import yaml
import glob
from tqdm import trange
from datetime import datetime 
import wandb
import torch 
import torch.nn as nn  
from torch.utils.data import DataLoader 
from torch.optim import SGD 
from sklearn.metrics import precision_recall_fscore_support

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18



def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader

def log_data_samples(dataLoader):
    table = wandb.Table(columns=["Image", "Label", "Filename"])
    
    for i, batch in enumerate(dataLoader):
        if i >= 10:  # Log only 10 samples
            break
        
        images, labels, filenames = batch  

        table.add_data(wandb.Image(images[0]), labels[0], filenames[0])  # Log first sample

    wandb.log({"Sample Data": table})

    
def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])    # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('model_states/*.pt')
    if len(model_states):
        # at least one save state found; get latest
        model_states = [model_state for model_state in model_states if not "last.pt" in model_state  and not "best.pt" in model_state ]
        model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch last.pt')
        state = torch.load(open(f'model_states/last.pt', 'rb'), map_location=cfg['device'])
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch


def save_model(cfg, epoch, model, stats):
    os.makedirs('model_states', exist_ok=True)

    # Save model state
    model_path = f'model_states/{epoch}.pt'
    stats['model'] = model.state_dict()
    torch.save(stats, open(model_path, 'wb'))

    # Log model checkpoint as a WandB artifact
    wandb.log({"epoch": epoch})  # Log epoch number
    artifact = wandb.Artifact(
        name=f"model_checkpoint_{epoch}",  # Unique artifact name per epoch
        type="model",
        metadata={"epoch": epoch}  # Extra info about this checkpoint
    )
    artifact.add_file(model_path)  # Add saved model file
    wandb.log_artifact(artifact)  # Upload artifact to WandB
    
    # Link the artifact to the WandB Model Registry
    run = wandb.run  # Get the current active run
    if run is not None:
        run.link_artifact(artifact, "sarah_dsi/wandb-registry-model/best_model")  
        # ^ Change "best_model" to the appropriate collection name

    # Also save config file if not present
    cfpath = 'model_states/configs_used_for_this_run.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)        

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer


def log_predictions_table(phase, model, dataLoader, cfg, max_samples=20):
    """
    Logs a table comparing ground truth labels with predictions.
    
    Args:
        phase (str): "train" or "validation"
        model: The trained model
        dataLoader: DataLoader for the dataset
        cfg: Configuration dictionary
        max_samples (int): Max number of samples to log
    """
    device = cfg['device']
    model.to(device)
    model.eval()  # Set model to evaluation mode

    table = wandb.Table(columns=["Image", "Ground Truth", "Prediction", "Filename"])
    logged_samples = 0

    with torch.no_grad():  
        for data, labels, image_names in dataLoader:
            data, labels = data.to(device), labels.to(device)

            # Get predictions
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            # Log a limited number of samples
            for i in range(len(data)):
                if logged_samples >= max_samples:
                    break
                table.add_data(
                    wandb.Image(data[i].cpu()),  # Convert tensor image to a WandB image
                    labels[i].item(),  # Ground truth
                    preds[i].item(),  # Model prediction
                    image_names[i]  # Filename
                )
                logged_samples += 1
            
            if logged_samples >= max_samples:
                break  # Stop if max samples reached

    # Log the table to WandB
    wandb.log({f"{phase.capitalize()} Predictions": table})


def train(cfg, dataLoader, model, optimizer):
    all_preds = []
    all_labels = []
    
    device = cfg['device'] 
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    loss_total, oa_total = 0.0, 0.0  

    progressBar = trange(len(dataLoader))
    for idx, (data, labels, image_names) in enumerate(dataLoader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        prediction = model(data)

        # Reset gradients
        optimizer.zero_grad()

        # Compute loss
        loss = criterion(prediction, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log statistics
        loss_total += loss.item()
        pred_label = torch.argmax(prediction, dim=1)    
        oa = torch.mean((pred_label == labels).float()) 
        oa_total += oa.item()
        
        all_preds.extend(pred_label.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)

    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    

    
    # Compute per-class precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=list(range(cfg['num_classes']))
)

    # # Log loss and accuracy to WandB
    # wandb.log({"Train Loss": loss_total, "Train Accuracy": oa_total, step=epoch})

    # Log to wandb for each class
    for i, (p, r, f1s) in enumerate(zip(precision, recall, f1)):
        wandb.log({
            f"Train Precision Class {i}": p,
            f"Train Recall Class {i}": r,
            f"Train F1-score Class {i}": f1s,
        })

    # 🔥 Log predictions for train phase 🔥
    log_predictions_table("train", model, dataLoader, cfg)

    return loss_total, oa_total, p, r, f1s
    

def validate(cfg, dataLoader, model):
    all_preds = []
    all_labels = []
    
    device = cfg['device']
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    loss_total, oa_total = 0.0, 0.0  

    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():
        for idx, (data, labels, image_names) in enumerate(dataLoader):
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            prediction = model(data)

            # Compute loss
            loss = criterion(prediction, labels)

            # Log statistics
            loss_total += loss.item()
            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)

    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    
    # Compute per-class precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=list(range(cfg['num_classes']))
)

    # # Log validation metrics to WandB
    # wandb.log({"Validation Loss": loss_total, "Validation Accuracy": oa_total, step=epoch})

    # Log to wandb for each class
    for i, (p, r, f1s) in enumerate(zip(precision, recall, f1)):
        wandb.log({
            f"Train Precision Class {i}": p,
            f"Train Recall Class {i}": r,
            f"Train F1-score Class {i}": f1s,
        })

    # 🔥 Log predictions for validation phase 🔥
    log_predictions_table("validation", model, dataLoader, cfg)

    return loss_total, oa_total, p, r, f1s


def parse_args():
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for training')
    return parser.parse_args()


def main():
    args = parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    # Override config with sweep arguments (if provided)
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.learning_rate:
        cfg['learning_rate'] = args.learning_rate
    if args.num_epochs:
        cfg['num_epochs'] = args.num_epochs
    if args.weight_decay:
        cfg['weight_decay'] = args.weight_decay
        
    wandb.login()

    wandb.init(
    project="cv4ecology",
    entity="catalyst_dsi",
    config=cfg,
    ) # set the wandb project where this run will be logged

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))
    
    device = cfg['device']
    
    # Check if the selected device is either 'cuda' or 'mps' and available; otherwise, fall back to 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA is not available; falling back to CPU...')
        cfg['device'] = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print(f'WARNING: device set to "{device}" but MPS is not available; falling back to CPU...')
        cfg['device'] = 'cpu'
    elif device not in ['cuda', 'mps', 'cpu']:
        print(f'WARNING: device set to "{device}" is invalid; falling back to CPU...')
        cfg['device'] = 'cpu'
        
    device = torch.device(cfg['device'])
    
    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
    
    log_data_samples(dl_train)
    log_data_samples(dl_val)
    
    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # Early stopping parameters
    patience = cfg.get('patience', 10)  # Number of epochs to wait for improvement, sets to 10 if not defined in config file
    print(f"Starting training with a patience value of {patience}") #useful info when running your model
    best_loss_val = float('inf')  # Best validation loss encountered
    epochs_without_improvement = 0  # Counter for patience

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train, p_train, r_train, f1s_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val, p_valid, r_valid, f1s_valid = validate(cfg, dl_val, model)

        # combine stats and save
        stats = {
            'Train Loss': loss_train,
            'Valid Loss': loss_val,
            'Train Overall Accuracy': oa_train,
            'Valid Overall Accuracy': oa_val,
            'Train Recall': r_train,
            'Valid Recall': r_recall,
            'Train Precision': p_train,
            'Valid Precision': p_valid,
            'Train F1': f1s_train,
            'Valid F1': f1s_valid,
            'epoch': epoch
        }

        # this is checkpoint saving, this saves all models
        save_model(cfg, current_epoch, model, stats)
        #cfg: config
        save_model(cfg, 'last', model, stats) #save last model

        # Early stopping logic
        if loss_val < best_loss_val:
            best_loss_val = loss_val  # Update the best validation loss
            epochs_without_improvement = 0  # Reset patience counter
            save_model(cfg, 'best', model, stats) #second argument into this function names the model results
            print(f"Best model!!!! saving model at epoch {current_epoch}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break  # Exit training loop
    
        # log metrics to wandb
        wandb.log(stats) #this takes a dict, stats is already a dict

    # Get current date and time
    now = datetime.now()

    # Format it as a string that can be safely used in a folder name
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    print(timestamp)  # Output will look like '2025-01-16_12-30-45'

    os.rename('model_states', f'model_states-{timestamp}')

    # That's all, folks!
    wandb.finish()
        

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()