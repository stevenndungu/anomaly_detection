import torch, random, os
import torch.nn as nn
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import preprocessing

def get_data(path, dic_labels, num_filters):
      # Load the MATLAB file
      data = loadmat(path)
      df0 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][0])
      df0['label'] = 'FRI'
      df1 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][1])
      df1['label'] = 'FRII'
      df_train = pd.concat([df0, df1], ignore_index=True)

      df0 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][0])
      df0['label'] = 'FRI'
      df1 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][1])
      df1['label'] = 'FRII'
      df2 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][2])
      df2['label'] = 'RRG'
      df3 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][3])
      df3['label'] = 'XRG'
      df_test = pd.concat([df0, df1, df2, df3], ignore_index=True)

      column_names = ["descrip_" + str(i) for i in range(1, df_train.shape[1])] + ["label_code"]
      df_train.columns = column_names
      df_test.columns = column_names

      df_train['label_code'] = df_train['label_code'].map(dic_labels)
      df_test['label_code'] = df_test['label_code'].map(dic_labels)

      #subset based number of fileters
      df_train = df_train[["descrip_" + str(i) for i in range(1, 2*num_filters+1)] + ["label_code"]]
      df_test = df_test[["descrip_" + str(i) for i in range(1, 2*num_filters+1)] + ["label_code"]]

      return df_train, df_test

# Function to evaluate the model and calculate metrics
def evaluate_model(y_true, y_pred):
   accuracy = accuracy_score(y_true, y_pred)
   precision = precision_score(y_true, y_pred, average='binary', zero_division=0.0)
   recall = recall_score(y_true, y_pred, average='binary', zero_division=0.0)
   f1 = f1_score(y_true, y_pred, average='binary', zero_division=0.0)
   gmean = geometric_mean_score(y_true, y_pred, average="binary")
   conf_matrix = confusion_matrix(y_true, y_pred,labels=[1, 0])
#    print(conf_matrix)
#    fig, ax = plt.subplots(figsize=(7.5, 7.5))
#    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
#    for i in range(conf_matrix.shape[0]):
#      for j in range(conf_matrix.shape[1]):
            
#             ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

#    plt.xlabel('Predictions', fontsize=18)
#    plt.ylabel('Actuals', fontsize=18)
#    plt.title('Confusion Matrix', fontsize=18)
#    plt.savefig('confusion_matrix.png')
#    plt.close()
   
   tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
   #print(tn, fp, fn, tp)
   specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
   f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
   return accuracy, precision, recall, f1, specificity, f2, gmean


class CosfireDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(preprocessing.normalize(dataframe.iloc[:, :-1].values), dtype=torch.float32)
        self.labels = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# def process_data_previous(path ,dic_labels,num_filters, random_state=42):
#    """
#    Process train and test data into train, test and validation sets
#    """
#    # Read data
#    df_train, df_test = get_data(path, dic_labels, num_filters)

   
#    # Sample validation data from train
#    frii_valid = df_train[df_train['label_code'] == 1].sample(n=53, random_state=random_state)
#    fri_valid = df_train[df_train['label_code'] == 0].sample(n=22, random_state=random_state)
   
#    # Get remaining train data
#    selected_indices = list(frii_valid.index) + list(fri_valid.index)
#    train_data = df_train.drop(index=selected_indices).reset_index(drop=True)
   
#    # Process test data by label
#    test_splits = {}
#    for label, code in {'FRII': 1, 'FRI': 0, 'XRG': 3, 'RRG': 2}.items():
#        test_splits[label] = df_test[df_test['label_code'] == code].reset_index(drop=True)
   
#    # Create final datasets
#    test_data = pd.concat([
#        test_splits['FRII'], 
#        test_splits['FRI'],
#        test_splits['XRG'].tail(22),
#        test_splits['RRG'].tail(13)
#    ])
   
#    valid_data = pd.concat([
#        test_splits['XRG'].head(22),
#        test_splits['RRG'].head(12),
#        frii_valid,
#        fri_valid
#    ]).reset_index(drop=True)
   
#    return train_data, test_data, valid_data


def process_data(path ,dic_labels,num_filters, random_state=42):
   """
   Process train and test data into train, test and validation sets
   """
   # Read data
   df_train, df_test = get_data(path, dic_labels, num_filters)

   
   # Sample validation data from train
   frii_valid = df_train[df_train['label_code'] == 1].sample(n=53, random_state=random_state)
   fri_valid = df_train[df_train['label_code'] == 0].sample(n=22, random_state=random_state)
   
   # Get remaining train data
   selected_indices = list(frii_valid.index) + list(fri_valid.index)
   train_data = df_train.drop(index=selected_indices).reset_index(drop=True)
   
   # Process test data by label
   test_splits = {}
   for label, code in {'FRII': 1, 'FRI': 0, 'XRG': 3, 'RRG': 2}.items():
       test_splits[label] = df_test[df_test['label_code'] == code].reset_index(drop=True)
   
   # Create final datasets
   test_data = pd.concat([
       test_splits['FRII'], 
       test_splits['FRI'],
       test_splits['XRG'],
       test_splits['RRG'].tail(13)
   ])
   
   valid_data = pd.concat([
       test_splits['RRG'].head(90),
       frii_valid,
       fri_valid
   ]).reset_index(drop=True)
   
   return train_data, test_data, valid_data


# def generate_hidden_dims(input_dim, reduction=50, min_neurons=10):
#     """
#     Generates a list of hidden dimensions by repeatedly subtracting 'reduction'
#     from the current dimension, but stops when the next subtraction would drop
#     below 'min_neurons'.
    
#     Args:
#         input_dim (int): The size of the input layer.
#         reduction (int): The number of neurons to subtract at each layer.
#         min_neurons (int): The minimum allowed number of neurons.
    
#     Returns:
#         List[int]: A list of hidden layer dimensions for the encoder.
#     """
#     dims = []
#     current_dim = input_dim
#     # Continue subtracting until the next layer would be less than min_neurons.
#     while current_dim - reduction >= min_neurons:
#         current_dim -= reduction
#         dims.append(current_dim)
#     return dims

# class AutoencoderStep(nn.Module):
#     def __init__(self, input_dim, reduction=50, min_neurons=50, dropout=0.0):
#         """
#         Constructs an autoencoder with hidden layers that reduce by a fixed number
#         of neurons (reduction) until reaching a minimum size (min_neurons).
        
#         For example, if input_dim=200, reduction=50, and min_neurons=10:
#           - Encoder layers will be: 200 -> 150 -> 100 -> 50
#           - Decoder layers will mirror the encoder: 50 -> 100 -> 150 -> 200
        
#         Args:
#             input_dim (int): Dimensionality of the input data.
#             reduction (int): Number of neurons to subtract for each subsequent hidden layer.
#             min_neurons (int): The minimum number of neurons allowed in the final encoder layer.
#             dropout (float): Dropout probability to use after activation (except the last encoder layer).
#         """
#         super(AutoencoderStep, self).__init__()
        
#         # Generate the list of hidden dimensions for the encoder.
#         # Example: for input_dim=200, reduction=50, min_neurons=10 -> [150, 100, 50]
#         hidden_dims = generate_hidden_dims(input_dim, reduction, min_neurons)
        
#         # Build the encoder
#         encoder_layers = []
#         prev_dim = input_dim
#         for i, h in enumerate(hidden_dims):
#             encoder_layers.append(nn.Linear(prev_dim, h))
#             # For all but the final encoder layer, add BatchNorm, ReLU, and Dropout.
#             if i < len(hidden_dims) - 1:
#                 encoder_layers.append(nn.BatchNorm1d(h))
#                 encoder_layers.append(nn.ReLU(inplace=True))
#                 encoder_layers.append(nn.Dropout(dropout))
#             prev_dim = h
        
#         # If no hidden dimensions were generated, the encoder is the identity.
#         self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        
#         # Build the decoder as the mirror image of the encoder.
#         decoder_layers = []
#         # Reverse the hidden dimensions for the decoder.
#         if hidden_dims:
#             rev_hidden_dims = list(reversed(hidden_dims))
#             prev_dim = rev_hidden_dims[0]
#             for i, h in enumerate(rev_hidden_dims[1:]):
#                 decoder_layers.append(nn.Linear(prev_dim, h))
#                 decoder_layers.append(nn.BatchNorm1d(h))
#                 decoder_layers.append(nn.ReLU(inplace=True))
#                 decoder_layers.append(nn.Dropout(dropout))
#                 prev_dim = h
#             # Final layer maps back to the original input dimension.
#             decoder_layers.append(nn.Linear(prev_dim, input_dim))
#         else:
#             # If no encoder layers exist, simply map input to input.
#             decoder_layers.append(nn.Linear(input_dim, input_dim))
        
#         # Assuming the input data is normalized between 0 and 1, we use Sigmoid.
#         decoder_layers.append(nn.Sigmoid())
#         self.decoder = nn.Sequential(*decoder_layers)
    
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


# For an input dimension of 200, with a step of 50, and a minimum layer size of 10,
# the encoder layers will be: 200 -> 150 -> 100 -> 50.
#model = AutoencoderStep(input_dim=100, reduction=30, min_neurons=10, dropout=0.1)

# Print the model architecture.
#print(model)


#For Reproducibility
def reproducibility_requirements(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print("Set seed of", str(seed),"is done for Reproducibility")




# class AutoencoderWithEmbeddings(nn.Module):
#     def __init__(self, input_dim=200, dpt=0, step_size=50, min_dim=10):
#         super().__init__()
        
#         # Calculate layer dimensions dynamically
#         layer_dims = self._calculate_layer_dims(input_dim, step_size, min_dim)
        
#         # Encoder
#         encoder_layers = []
#         for i in range(len(layer_dims) - 1):
#             encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
#             if i < len(layer_dims) - 2:  # Only apply for layers before the final encoder layer
#                 encoder_layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
#                 encoder_layers.append(nn.ReLU())
#                 encoder_layers.append(nn.Dropout(dpt))
#         self.encoder = nn.Sequential(*encoder_layers)
        
#         # Decoder (mirror of encoder)
#         decoder_layers = []
#         for i in range(len(layer_dims) - 1, 0, -1):
#             decoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i - 1]))
#             decoder_layers.append(nn.BatchNorm1d(layer_dims[i - 1]))
#             decoder_layers.append(nn.ReLU())
#             decoder_layers.append(nn.Dropout(dpt))
#         decoder_layers.append(nn.Sigmoid())  # Final activation for reconstruction
#         self.decoder = nn.Sequential(*decoder_layers)

#     def _calculate_layer_dims(self, input_dim, step_size, min_dim):
#         """
#         Dynamically calculate layer dimensions, reducing by `step_size` at each layer
#         until the minimum dimension (`min_dim`) is reached.
#         """
#         layer_dims = [input_dim]
#         current_dim = input_dim
        
#         while current_dim - step_size >= min_dim:
#             current_dim -= step_size
#             layer_dims.append(current_dim)
        
#         # Ensure the last layer is at least `min_dim`
#         if layer_dims[-1] != min_dim:
#             layer_dims.append(min_dim)
        
#         return layer_dims

#     def forward(self, data):
#         encoded = self.encoder(data)
#         decoded = self.decoder(encoded)
#         return decoded



class AutoencoderWithEmbeddings(nn.Module):
    def __init__(self, input_dim=200, dpt=0):
        super().__init__()
        # Encoder
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            # nn.BatchNorm1d(150),
            # nn.ReLU(),
            # nn.Dropout(dpt),
            # nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(dpt),
            nn.Linear(100, 50),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(dpt),
            # nn.Linear(100, 150),
            # nn.BatchNorm1d(150),
            # nn.ReLU(),
            # nn.Dropout(dpt),
            nn.Linear(100, input_dim),
            nn.Sigmoid()
        )

    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
    
   
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path=f'auto_encoder_results/checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time the validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'auto_encoder_results/checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss 

        if self.best_score is None:
            self.best_score = score 
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
   


def predict(model: nn.Module, loader: DataLoader, device: torch.device): 
    reconstruction_errors = []
    for data in loader:
        # Move inputs to the correct device (if not handled in Dataset)
        data = data[0].to(device)

        # Forward pass
        outputs = model(data)

        # Compute reconstruction errors for the batch
        error = torch.mean((outputs - data) ** 2, dim=1).cpu()  # Ensure it's on CPU before converting to NumPy
        reconstruction_errors.append(error)
    
    # Concatenate all batch-wise errors and return as a Pandas Series
    return pd.Series(torch.cat(reconstruction_errors).numpy(), name="prediction")

