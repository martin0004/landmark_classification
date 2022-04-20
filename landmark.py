################################################################################
#                                                                              #
# IMPORTS                                                                      #
#                                                                              #
################################################################################

# Standard libairies

import json
from typing import Dict, List, Union

# Third-party librairies

import numpy as np
import PIL
import plotly.graph_objects as go
import plotly.subplots as psub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import *

################################################################################
#                                                                              #
# DATALOADERS                                                                  #
#                                                                              #
################################################################################

def get_data_loaders(dir_train: str,
                     dir_valid: Union[str,None],
                     dir_test: str,
                     ratio: Union[float,None] = None) -> Dict:
    """Build train-valid-test DataLoaders from ImageFolders.
    
    Inputs
    
    dir_train         Path to train directory.
    dir_valid         Path to valid directory.
    dir_test          Path to test directory.

    Optional arguments
    
    ratio             Ratio of valid images over train images.
                      (Use if only train-test datasets available
                      and no valid dataset. Specify dir_valid=None
                      if using this ratio to create valid dataset.)
    
    Outputs
    
    dls               Train-valid-test DataLoaders. 
    
    """
    
    # Means and standard deviations for normalization
    # https://pytorch.org/vision/stable/models.html
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    
    # Number of pixels for resizing/cropping images
    px_resize = 255
    px_crop = 224
    
    # Angle for rotations
    deg = 30
    
    # Train loader
    
    tr_train = Compose( [RandomRotation(deg),
                         RandomResizedCrop(px_crop),
                         RandomHorizontalFlip(),
                         ToTensor(),
                         Normalize(means, stds)] )
    
    dt_train = ImageFolder(dir_train, transform=tr_train)
    dl_train = DataLoader(dt_train, batch_size=64)

    # Valid loader
    
    tr_valid = Compose ( [Resize(px_resize),
                          CenterCrop(px_crop),
                          ToTensor(),
                          Normalize(means, stds)] )
    
    # ratio not specified = valid_dataset created from dir_valid.
    if ratio is None:    
        
        dt_valid = ImageFolder(dir_valid, transform = tr_valid)
        dl_valid = DataLoader(dt_valid, batch_size=64)
    
    # radio specified = valid_dataset created from train_dataset
    else:
        
        # Indices of train-valid items
        n = len(dt_train)
        idx = np.array(range(n))
        np.random.shuffle(idx)
        
        # Split
        n_train = int( (1 - ratio)*n )
        idx_train = idx[:n_train]
        idx_valid = idx[n_train:]
        sampler_train = SubsetRandomSampler(idx_train)
        sampler_valid = SubsetRandomSampler(idx_valid)
        
        # Update train-valid loaders
        dl_train = DataLoader(dt_train, batch_size=64, sampler=sampler_train)
        dt_valid = ImageFolder(dir_train, transform = tr_valid)
        dl_valid = DataLoader(dt_valid, batch_size=64, sampler=sampler_valid)
    
    # Test loader
    
    tr_test = tr_valid
    dt_test = ImageFolder(dir_test, transform = tr_test)
    dl_test = DataLoader(dt_test, batch_size=64)
    
    # Return dataloaders
        
    dls = dict()
    dls["train"] = dl_train
    dls["valid"] = dl_valid
    dls["test"]  = dl_test
    
    return dls


################################################################################
#                                                                              #
# NEURAL NETWORKS                                                              #
#                                                                              #
################################################################################

class NeuralNetworkScratch(nn.Module):
    """Convolutional neural network built "from scratch" in this project.
    """
    
    def __init__(self):
        
        super(NeuralNetworkScratch, self).__init__()
        
        # Conv stack 1
        # Input: 224 x 224 x 3
        # Output: 112 x 112 x 16
        self.conv_1_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv_1_2 = nn.Conv2d(16, 16, 3, padding=1)
        
        # Conv stack 2
        # Input: 112 x 112 x 16
        # Output: 56 x 56 x 32
        self.conv_2_1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv_2_2 = nn.Conv2d(32, 32, 3, padding=1)
        
        # Conv stack 3
        # Input: 56 x 56 x 32
        # Output: 28 x 28 x 64
        self.conv_3_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_3_2 = nn.Conv2d(64, 64, 3, padding=1)     

        # Max pooling layers
        self.max_pool = nn.MaxPool2d(2)
        
        # Fully connected layer 1
        # Input: 50176 (28*28*64)
        # Output: 1000
        self.fc_1 = nn.Linear(50176, 1000)
        
        # Fully connected layer 2
        # Input: 1000
        # Output: 50 (number of classes)
        self.fc_2 = nn.Linear(1000, 50)
      
        # Activation functions        
        self.relu = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(0.20)
        
        # Output layer
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        
        # Conv stack 1
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.max_pool(x)
        
        #x = self.dropout(x)
        
        # Conv stack 2
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.max_pool(x)
        
        # Conv stack 3
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.max_pool(x)
        
        # Flatten layer
        x = x.flatten(1)
        x = self.dropout(x)
        
        # Fully connected layer 1
        x = self.fc_1(x)
        x = self.relu(x)

        # Fully connected layer 2
        x = self.fc_2(x)
        x = self.relu(x)
        
        # Output layer        
        x = self.log_softmax(x)
        
        return x


def get_model_scratch() -> nn.Module:
    """Instantiate the CNN built 'from scratch' in this project.
    
    Output
    
    model        Neural network (untrained).
    
    """
    
    # Instantiate model
    model = NeuralNetworkScratch()
    
    return model


def get_model_transfer() -> nn.Sequential:
    """Instantiate the CNN built from transfer learning in this project.
                 
    Output
    
    model        Neural network (untrained).
    
    """
    
    
    # Import pre-trained model
    model = torchvision.models.vgg16(pretrained=True)
        
    # Number of classifier input features
    n_input = model.classifier[0].in_features
    
    # New classifier
    # (reusing same architecture as scratch model)   
    model.classifier = nn.Sequential(nn.Linear(n_input, 1000), # fc1
                                     nn.ReLU(),
                                     nn.Linear(1000, 50),      # fc2
                                     nn.ReLU(),
                                     nn.LogSoftmax(dim=1)      # output
                                    )
    
    return model


################################################################################
#                                                                              #
# TRAINING / TESTING                                                           #
#                                                                              #
################################################################################

def train(n_epochs: int,
          dls: Dict,
          model: nn.Module,
          optimizer,
          criterion,
          file_checkpoint: str,
          file_metrics: str,
          device: str = "cpu") -> None:
    """Train neural network.
    
    Inputs
    
    n_epochs            Number of training epochs.
    dls                 DataLoaders.
    model               Neural Network.
    optimizer           Model optimizer.
    criterion           Model loss function.
    file_checkpoint     File containing model checkpoint.
    file_metrics        File containing model metrics.
    device              Device for running calculations.
                        ("cpu" or "gpu")
    
    """
    
    # INITIALIZE
    
    # Move model to CPU or GPU
    model.to(device)
    
    # Lowest valid loss
    valid_loss_min = np.Inf 
    
    # Number of mini_batches
    n_mini_batch_train = len(dls["train"])
    n_mini_batch_valid = len(dls["valid"])
    
    # Number of data points
    n_train = len(dls["train"].sampler)
    n_valid = len(dls["valid"].sampler)
    
    # Metrics dictionnary
    d_metrics = dict()
    d_metrics["epoch"] = []
    d_metrics["train_loss"] = []
    d_metrics["train_acc"] = []
    d_metrics["valid_loss"] = []
    d_metrics["valid_acc"] = []
    d_metrics["checkpoint"] = []
    
    for epoch in range(1, n_epochs+1):
        
        # Initialize loss
        # (mean cross-entropy loss for a single point - over entire dataset)
        train_loss_dt = 0.0    
        valid_loss_dt = 0.0
        
        # Initialiaze accuracy
        # (over entire dataset)
        train_acc_dt = 0.0
        valid_acc_dt = 0.0

        # TRAIN
        
        # Enable dropout
        model.train()
        
        for data, targets in dls['train']:
            
            # Move to CPU or GPU
            data = data.to(device)
            targets = targets.to(device)
            
            # Reset optimizer            
            optimizer.zero_grad()
            
            # Make predictions
            predictions_log = model(data)
            
            # Update loss
            # (mean cross-entropy loss for a single point - over mini-batch)
            loss_mini_batch = criterion(predictions_log, targets)
            loss_mini_batch.backward()            
            train_loss_dt += loss_mini_batch.item() / n_mini_batch_train
            
            # Update # correct predictions
            predictions_prob = torch.exp(predictions_log)
            predictions_idx = predictions_prob.topk(1, dim=1)[1].flatten()
            correct_mini_batch = torch.sum(predictions_idx == targets)
            train_acc_dt += correct_mini_batch.item() / n_train
            
            # Update weights            
            optimizer.step()
            
        # VALID
        
        # Disable gradient calculations
        with torch.no_grad():
        
            # Disable dropout
            model.eval()
        
            for data, targets in dls['valid']:
            
                # Move to CPU or GPU
                data = data.to(device)
                targets = targets.to(device)

                # Make predictions
                predictions_log = model(data)
            
                # Update loss
                loss_mini_batch = criterion(predictions_log, targets) # average over mini-batch             
                valid_loss_dt += loss_mini_batch.item() / n_mini_batch_valid
                
                # Update accuracy
                predictions_prob = torch.exp(predictions_log)
                predictions_idx = predictions_prob.topk(1, dim=1)[1].flatten()
                correct_mini_batch = torch.sum(predictions_idx == targets)
                valid_acc_dt += correct_mini_batch.item() / n_valid

        # PRINT
        
        # Metrics
        print("Epoch:"       , epoch, end = "")
        print("    Training acc:"  , format(train_acc_dt,".3f"), end="") 
        print("    Training loss:" , format(train_loss_dt,".3f"), end="") 
        print("    Validation acc:"  , format(valid_acc_dt,".3f"), end="") 
        print("    Validation loss:" , format(valid_loss_dt,".3f")) 
        
        # SAVE
        
        # Model metrics (every epoch)        
        d_metrics["epoch"].append(epoch)
        d_metrics["train_loss"].append(train_loss_dt)
        d_metrics["train_acc"].append(train_acc_dt)
        d_metrics["valid_loss"].append(valid_loss_dt)
        d_metrics["valid_acc"].append(valid_acc_dt)
        
        if valid_loss_dt < valid_loss_min:
            # 1 = This epoch is the latest checkpoint.
            # 0 = All other epochs are not latest checkpoint.
            d_metrics["checkpoint"] = [0 for epoch in d_metrics["epoch"]]
            d_metrics["checkpoint"][-1] = 1
        else:
            # This epoch is not the latest checkpoint -> Append 0.
            d_metrics["checkpoint"].append(0)
            
        with open(file_metrics, "w") as f:
            json.dump(d_metrics, f)
        
        # Model checkpoint (only when validation loss decreases)        
        if valid_loss_dt < valid_loss_min:
            print("Valid loss decreased - saving model to file.")
            valid_loss_min = valid_loss_dt
            torch.save(model.state_dict(), file_checkpoint)

def test(model: nn.Module,
         dls: Dict,
         criterion,
         device: str = "cpu") -> None:
    """Test neural network.
    
    Inputs
    
    model               Neural Network.
    dls                 DataLoaders.
    criterion           Model loss function.
    device              Device for running calculations.
                        ("cpu" or "gpu")
    
    """
    
    # INITIALIZE
    
    # Move model to CPU or GPU
    model.to(device)
    
    # Number of mini_batches
    n_mini_batch_test = len(dls["test"])
    
    # Number of data points
    n_test = len(dls["test"].dataset)
    
    # Initialiaze accuracy
    # (over entire dataset)
    test_acc_dt = 0.0

    # Initialize loss
    # (mean cross-entropy loss for a single point - over entire dataset)
    test_loss_dt = 0.0    
        
    # Initialiaze accuracy
    # (over entire dataset)
    test_acc_dt = 0.0
    
    # TEST
        
    # Disable gradient calculations
    with torch.no_grad():
        
        # Disable dropout
        model.eval()
        
        for data, targets in dls['test']:
            
            # Move to CPU or GPU
            data = data.to(device)
            targets = targets.to(device)

            # Make predictions
            predictions_log = model(data)
            
            # Update loss
            loss_mini_batch = criterion(predictions_log, targets) # average over mini-batch         
            test_loss_dt += loss_mini_batch.item() / n_mini_batch_test
                
            # Update accuracy
            predictions_prob = torch.exp(predictions_log)
            predictions_idx = predictions_prob.topk(1, dim=1)[1].flatten()
            correct_mini_batch = torch.sum(predictions_idx == targets)
            test_acc_dt += correct_mini_batch.item() / n_test
        
    # PRINT
        
    # Metrics
    print("Test acc:"  , format(test_acc_dt,".3f"), end="") 
    print("    Test loss:" , format(test_loss_dt,".3f"))            
            

        
################################################################################
#                                                                              #
# METRICS                                                                      #
#                                                                              #
################################################################################   

def plot_metrics_file(file_metrics: str) -> None:
    """Plot model learning curve from metrics file.
    
    Input
    
    file_metrics        File containing model metrics.
    
    """
    
    # Load metrics
    
    with open(file_metrics, "r") as f:
        d_metrics = json.load(f)
    
    # Figure parameters
    
    n_rows = 1
    n_cols = 2
    
    y_min_acc = 0
    y_max_acc = 1
    
    y_min_loss = 0
    y_max_loss = 1.1*max(max(d_metrics["train_loss"]), max(d_metrics["valid_loss"]))

    
    # Subplot titles
    
    subplot_titles = ["Accuracy", "Loss"]   
    
    # Create figure
    
    fig = psub.make_subplots(rows=n_rows, cols=n_cols, subplot_titles = subplot_titles)
    
    # Plot accuracy
    
    x = d_metrics["epoch"]
    y = d_metrics["train_acc"]
    trace = go.Scatter(x=x, y=y, mode="lines", name="train", line_color = "green")
    fig.add_trace(trace, row=1, col=1)  
    
    x = d_metrics["epoch"]
    y = d_metrics["valid_acc"]
    trace = go.Scatter(x=x, y=y, mode="lines", name="valid", line_color = "blue")
    fig.add_trace(trace, row=1, col=1)      
    
    fig.layout.xaxis1.title = "epoch"
    fig.layout.yaxis1.title = "accuracy"    
    fig.layout.yaxis1.range = [y_min_acc, y_max_acc]

    # Plot loss
    
    x = d_metrics["epoch"]
    y = d_metrics["train_loss"]
    trace = go.Scatter(x=x, y=y, mode="lines", name="train", line_color = "green")
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)  
    fig.layout.xaxis2.title = "epoch"
    
    x = d_metrics["epoch"]
    y = d_metrics["valid_loss"]
    trace = go.Scatter(x=x, y=y, mode="lines", name="valid", line_color = "blue")
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)  
    
    fig.layout.xaxis2.title = "epoch"
    fig.layout.yaxis2.title = "loss"    
    fig.layout.yaxis2.range = [y_min_loss, y_max_loss]
    
    # Plot checkpoint
    
    idx = np.argmax(d_metrics["checkpoint"])
    epoch = d_metrics["epoch"][idx]
    
    valid_acc = d_metrics["valid_acc"][idx]
    trace = go.Scatter(x=[epoch], y=[valid_acc], mode="markers",
                       name="checkpoint", marker_color="black", marker_size=10)
    fig.add_trace(trace, row=1, col=1)
    
    valid_loss = d_metrics["valid_loss"][idx]
    trace = go.Scatter(x=[epoch], y=[valid_loss], mode="markers",
                       name="checkpoint", marker_color="black", marker_size=10)
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)
    
    fig.show()          
        
################################################################################
#                                                                              #
# UTILITY FUNCTIONS                                                            #
#                                                                              #
################################################################################

def get_prediction_probs(model: nn.Module,
                         dt: ImageFolder,
                         idx_img: int,
                         device: str="cpu") -> torch.Tensor:
    """Get neural network predictions (probabilities) for
    a single image from a DataLoader.
    
    Input
    
    model              Neural network
                       (output layer must be torch.nn.LogSoftmax())
    dt                 ImageFolder containing images for predictions.
    idx_img            Index of image for predictions.
    device             Device for running calculations.
                       ("cpu" or "gpu")
                 
    Output
    
    prediction_probs    Predictions (probabilities).
    
    """
    
    # Move model to CPU or GPU
    model.to(device)
    
    # Load image    
    image, _ = dt[idx_img]         # Get image (DxHxW)
    images = image.unsqueeze(0)    # Store image in input tensor (1xDxHxW)
    images = images.to(device)
    
    # Make predictions
    prediction_logs = model(images)
    prediction_probs = torch.exp(prediction_logs)

    return prediction_probs


def get_top_k_categories(model: nn.Module,
                         img_path: str,
                         d_categories: Dict,
                         k: int = 5,
                         device: str="cpu") -> List[str]:
    """Make predictions with a neural network and return
    k most likely categories.
    
    Input
    
    model                Neural network.
                         (output layer must be torch.nn.LogSoftmax())
    img_path             Path to image for predictions.
    d_categories         Dictionnary with category IDs/labels.
    k                    Number of most likely predictions to return.
    device               Device for running calculations.
                         ("cpu" or "gpu")
    
    Output
    
    top_k_categories     k most likely category labels.
    
    """
    
    # Move model to CPU or GPU
    model.to(device)
    
    # Load image
    pil = PIL.Image.open(img_path)
    
    # Convert to RGB
    rgb = pil.convert("RGB")
    
    # Pre-process image
    # (same transforms as test loader)

    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    px_resize = 255
    px_crop = 224
    deg = 30
    
    tr = Compose( [RandomRotation(deg),
                   RandomResizedCrop(px_crop),
                   RandomHorizontalFlip(),
                   ToTensor(),
                   Normalize(means, stds)] )
    
    rgb = tr(rgb)
    
    # Add extra dimension at beginning of tensor
    # so neural network can manipulate it.
    rgbs = rgb.unsqueeze(0)
        
    # Predictions
    model.eval()
    rgbs = rgbs.to(device)
    predictions_logs = model(rgbs)
    predictions_probs = torch.exp(predictions_logs)
    
    # Indexes of k highest probabilities
    _, idx_k = torch.topk(predictions_probs, k, dim=1)
    
    # Select only indices of 1st image
    idx_k = idx_k[0]
    
    # Classes of k highest probabilities
    top_k_categories = list()
    for idx in idx_k:
        category = d_categories[str(idx.item())]
        top_k_categories.append(category)
    
    # Return results
    return top_k_categories


def plot_top_k_categories(model: nn.Module,
                          img_path: str,
                          d_categories: Dict,
                          k: int = 5,
                          device: str="cpu") -> None:
    """Make predictions with a neural network and plot
    k most likely categories.
    
    Input
    
    model                Neural network.
                         (output layer must be torch.nn.LogSoftmax())
    img_path             Path to image for predictions.
    d_categories         Dictionnary with category IDs/labels.
    k                    Number of most likely predictions to return.
    device               Device for running calculations.
                         ("cpu" or "gpu")
    
    """
    
    # Top k predictions
    top_k_categories = get_top_k_categories(model, img_path, d_categories, k, device)

    # Figure parameters
    
    n_rows = 1
    n_cols = 2
    
    figure_height = 500
    figure_width = n_cols * figure_height
    
    # Subplots
    
    specs=[ [ {"type": "image"}, {"type": "table"} ] ]
    
    fig = psub.make_subplots(rows=n_rows, cols=n_cols, specs=specs)
    
    rgb = PIL.Image.open(img_path).convert("RGB")
    trace = go.Image(z=rgb)
    fig.add_trace(trace, row=1, col=1)
    fig.layout.xaxis1.showticklabels = False
    fig.layout.yaxis1.showticklabels = False
    
    table_header = {"values": ["Is this a picture of..."]}
    table_col_1 = [category + "?" for category in top_k_categories]
    table_cells = {"values": [table_col_1]}
    trace = go.Table(header=table_header, cells=table_cells)
    trace.columnwidth = [1, 5] # Relative widths
    trace.header.font.size = 20
    trace.header.height = 30
    trace.cells.font.size = 20
    trace.cells.height = 30
    fig.add_trace(trace, row=1, col=2)
    
    # Figure
    
    fig.layout.height = figure_height
    fig.layout.width = figure_width
    
    fig.show()        

    
    