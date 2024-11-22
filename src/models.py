from pathlib import Path 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from tqdm import tqdm  
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
import groupers 
import json 
'''
The core machine learning pipeline. 

For a struct, if it has F fields, and suppose we take L hottest loops, there will be C groupings, 
Each struct will produce a F x L matrix, and F x 1 groupings, 
we have F x (L+1) after concat grouping vector to the feature vector, now the feature matrix takes both the grouping information and the feature vectors 
Then we generate this for all groupers, so we can get a 
C x F x (L+1) matrix 
we use this to predict a C x 1 one-hot encoding, indicating which grouping will be the best. 

We will use an encoder to encoder each F x (L+1) matrix into (hidden, ) as encoding from each group should not interfere
Then we concatenate them horizontally into (C * hidden, ) encoding 
Finally we pass this into a linear layer for classification, converting (B, C*hidden) into (B, C) 
''' 
HOTNESS_LOOP_CNT = 10 # consider 10 hottest loops 
DEFAULT_HIDDEN_SIZE = 64 
LOG_EPOCH_INTERVAL = 10 
DEFAULT_MODEL_PATH = "model.pth" 

# # Define a model to process each matrix of size F x (L+1)
# class MatrixEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, rnn_type="GRU"):
#         super(MatrixEncoder, self).__init__()
#         # You can use LSTM, GRU, or even Transformer layers
#         self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
#         # (B, seqlen, input) --> 
#         # output: (B, seqlen, input) 
#         # hidden: (num layers, B, hidden)
#         self.fc = nn.Linear(hidden_size, hidden_size)  # Final encoding layer
#         # will be (*, input) --> (*, hidden)

#     def forward(self, x):
#         # print("Encoder input: ", x.shape) 
#         # x shape will be (B, C, F, L+1), need to reshape to (B*C, F, L+1) for separate processing 
#         x = x.view(-1, x.shape[2], x.shape[3]) 
#         # print("Reshaped x: ", x.shape) 
        
#         # x: (batch_size, F, L+1), batch of matrices
#         _, h_n = self.rnn(x)  # h_n will be of shape (1, batch_size, hidden_size)
#         return self.fc(h_n.squeeze(0))  # Output a fixed-size encoding
#         # output size: [B, hidden] 



class MatrixEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="GRU"):
        """
        A flexible encoder that supports GRU, LSTM, or RNN.
        
        Args:
        - input_size (int): Input feature size.
        - hidden_size (int): Hidden state size of the RNN.
        - rnn_type (str): Type of RNN to use ('GRU', 'LSTM', or 'RNN').
        """
        super(MatrixEncoder, self).__init__()
        
        # Select the appropriate RNN type
        rnn_mapping = {
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
            "RNN": nn.RNN
        }
        if rnn_type not in rnn_mapping:
            raise ValueError(f"Invalid rnn_type: {rnn_type}. Choose from 'GRU', 'LSTM', or 'RNN'.")
        
        self.rnn_type = rnn_type
        self.rnn = rnn_mapping[rnn_type](input_size=input_size, hidden_size=hidden_size, 
                                          num_layers=1, batch_first=True)
        print("Encoder RNN: ", self.rnn) 
        # Fully connected layer to project hidden state to encoding
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (B, C, F, L+1).
        
        Returns:
        - torch.Tensor: Encoded output of shape (B, hidden_size).
        """
        # Reshape input to (B*C, F, L+1) for processing
        x = x.view(-1, x.shape[2], x.shape[3])
        
        # Pass through the RNN
        if self.rnn_type == "LSTM":
            _, (h_n, _) = self.rnn(x)  # For LSTM, h_n is returned along with c_n
        else:
            _, h_n = self.rnn(x)  # For GRU or RNN, only h_n is returned
        
        # Project the hidden state to a fixed-size encoding
        return self.fc(h_n.squeeze(0))
   
   # Define a model that combines the C matrix encodings and predicts the best matrix
class BestMatrixSelector(nn.Module):
    def __init__(self, input_size, hidden_size, C, rnn_type):
        super(BestMatrixSelector, self).__init__()
        self.encoder = MatrixEncoder(input_size, hidden_size, rnn_type)  # To encode each matrix
        self.fc = nn.Linear(C * hidden_size, C)  # Classifier to select the best matrix

    def forward(self, matrices):
        # print("matrices: ", matrices.shape)
        B = matrices.shape[0]  
        # matrices: List of C matrices of shape (F, L+1)
        # encoded_matrices = [self.encoder(matrix) for matrix in matrices]
        encoded = self.encoder(matrices) 
        # print("encoded result: ", encoded.shape) 
        # result will be (B*C, hidden) 
        # this needs to be reshaped into (B, C*hidden) 
        encoded = encoded.view(B, -1) 
        # print("Reshaped encoded result: ", encoded.shape) 
        
        # Concatenate the encoded matrices into a single vector of size C * hidden_size
        # combined_encoding = torch.cat(encoded_matrices, dim=-1)
        # print("combined encoding: ", combined_encoding.shape)
        # Pass through the final layer to predict the best matrix
        return self.fc(encoded)

class GroupingSelector: 
    def __init__(self, C, L = HOTNESS_LOOP_CNT, hidden_size = DEFAULT_HIDDEN_SIZE, rnn_type="GRU"):
        ''' 
        C: number of grouping methods 
        L: L hottest loops 
        ''' 
        self.input_size = L+1 
        self.model = BestMatrixSelector(input_size=self.input_size, hidden_size=hidden_size, C=C, rnn_type=rnn_type)  
    def train(self, ds_df:pd.DataFrame, epochs = 100): 
        print("Training Matrix Selector model!!!") 
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in tqdm(range(epochs)): 
            for idx, row in ds_df.iterrows():  
                feature_grouping_matrices = row["feature_grouping_matrices"] 
                target = row["target"] 
                print(type(feature_grouping_matrices), type(target))
                print(feature_grouping_matrices.shape)  
                feature_grouping_matrices = torch.tensor(feature_grouping_matrices, dtype=torch.float32)  
                target = torch.tensor(target).long()  
                # print(target) 
                # print(feature_grouping_matrices) 
                
                # unsqueeze out the batch dimension 
                target = target.unsqueeze(0) 
                feature_grouping_matrices = feature_grouping_matrices.unsqueeze(0) 
                # print(target.shape, feature_grouping_matrices.shape) 

                output = self.model(feature_grouping_matrices) 
                
                loss = criterion(output, target) 
                
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 
                
                if (epoch+1) % LOG_EPOCH_INTERVAL == 0: 
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
    
    def test(self, ds_df: pd.DataFrame): 
        with torch.no_grad(): 
            all_preds = []
            all_targets = []
            all_probs = []  # List to store the predicted probabilities
            total_test_cnt = 0 
            total_correct_cnt = 0 

            for idx, row in ds_df.iterrows(): 
                # Extract the target and feature grouping matrices from the row
                target = row["target"] 
                feature_grouping_matrices = row["feature_grouping_matrices"] 
                target = torch.tensor(target).long().unsqueeze(0) 
                feature_grouping_matrices = torch.tensor(feature_grouping_matrices, dtype=torch.float32).unsqueeze(0)
                
                # Forward pass through the model
                output = self.model(feature_grouping_matrices) 
                
                # Get the predicted class (argmax)
                prediction = torch.argmax(output, dim=1)
                
                # Get the predicted probabilities using softmax (convert logits to probabilities)
                probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()  # Convert to numpy array
                
                # Collect predictions, probabilities, and targets for later evaluation
                all_preds.append(prediction.item())
                all_targets.append(target.item())
                all_probs.append(probs)  # Append the probabilities for each class
                
                # Calculate the test count and correct count
                test_cnt = prediction.numel()
                correct_cnt = (prediction == target).sum()
                total_test_cnt += test_cnt 
                total_correct_cnt += correct_cnt 

            # Calculate total accuracy
            total_correct_rate = total_correct_cnt / total_test_cnt 
            print(f"Total test count: {total_test_cnt}, Total correct count: {total_correct_cnt}, Total correct rate: {total_correct_rate}")

            # Convert the list of probabilities into a numpy array (shape: [num_samples, num_classes])
            all_probs = np.array(all_probs)

            # Return the predictions, targets, and probabilities
            return np.array(all_preds), np.array(all_targets), all_probs

    def get_conf_matrix(self, preds, targets, output_dir:Path): 
        cm = confusion_matrix(targets, preds)  
        fig, ax = plt.subplots() 
        cax = ax.matshow(cm, cmap=plt.cm.Blues, interpolation="nearest") 
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04) 
        cbar.set_label("Frequency", rotation=270, labelpad=10) 
        for (i, j), z in np.ndenumerate(cm): 
            ax.text(j,i,z,ha="center", va="center")
        grouper_names = groupers.get_all_grouper_names() 
        labels = "" 
        for idx, name in enumerate(grouper_names): 
            labels += f"{idx} : {name} \n" 
          
        plt.gcf().text(0.02, 0.4, labels, fontsize=9) 
        plt.subplots_adjust(left=0.5) 
        ax.set_xlabel("Predictions") 
        ax.xaxis.set_label_position("top") 
        ax.set_ylabel("True Labels") 
        plt.savefig( output_dir.joinpath("confusion_matrix.png"))  
        plt.close() 

    def plot_roc_curve(self, preds, targets, output_dir:Path):
        # Binarize the targets for multi-class ROC curve
        lb = LabelBinarizer()
        targets_bin = lb.fit_transform(targets)  # Convert to binary labels (one-hot encoding)
        # print("targets_bin: ", targets_bin) 
        
        # Get the class probabilities (preds should be probabilities, not hard predictions)
        # Example: for a model like Logistic Regression, preds can be the probabilities outputted by the model.
        # If preds contains hard class labels, you need to get probabilities from your model.
        preds_prob = preds  # Assuming preds contains the probability for each class (shape: [n_samples, n_classes])
        
        # Calculate ROC curve for each class
        n_classes = targets_bin.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], preds_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot the ROC curves for each class
        plt.figure(figsize=(10, 8))

        # Plot the ROC curve for each class
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {lb.classes_[i]} (AUC = {roc_auc[i]:.2f})')

        # Plot the diagonal line (no-skill line)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

        # Labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curve')
        plt.legend(loc='lower right')

        # Save the figure to a file
        output_file = output_dir.joinpath("roc_curve.png")
        plt.savefig(output_file)
        print(f"ROC curve saved as {str(output_file)}")
        plt.close()  # Close the plot to release resources

        # Calculate and print the overall AUC (macro average)
        overall_auc = np.mean(list(roc_auc.values()))
        print(f"Overall Macro Average AUC: {overall_auc:.2f}")
        return overall_auc 

    def evaluate(self, ds_df: pd.DataFrame, output_dir:Path=Path(".")):
        # Get predictions and targets
        preds, targets, probs = self.test(ds_df)
        print(f"preds: {preds} \n targets: {targets} \n probs: {probs}")

        # Calculate metrics
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        self.get_conf_matrix(preds=preds, targets=targets, output_dir=output_dir) 

        overall_auc = self.plot_roc_curve(preds=probs, targets=targets, output_dir=output_dir) 

        
        # Create a dictionary for metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1, 
            "overall_auc": overall_auc,
        }

        # Save metrics to a JSON file
        metrics_file_path = output_dir / "evaluation_metrics.json"
        try:
            with metrics_file_path.open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=4)
            print(f"Metrics saved to {metrics_file_path}")
        except Exception as e:
            print(f"Failed to save metrics to JSON: {e}")
        

    def model_output_to_probs(self, ds_df: pd.DataFrame):
        """Convert model outputs to probabilities (for ROC and AUC)"""
        all_probs = []
        with torch.no_grad():
            for idx, row in ds_df.iterrows(): 
                feature_grouping_matrices = row["feature_grouping_matrices"] 
                feature_grouping_matrices = torch.tensor(feature_grouping_matrices, dtype=torch.float32).unsqueeze(0)
                output = self.model(feature_grouping_matrices) 
                probs = torch.nn.functional.softmax(output, dim=1)
                all_probs.append(probs.numpy())
        return np.array(all_probs)
    
    def predict(self, feature_grouping_matrices):
        feature_grouping_matrices = torch.tensor(feature_grouping_matrices, dtype=torch.float32).unsqueeze(0)
        output = self.model(feature_grouping_matrices) 
        prediction = torch.argmax(output, dim=1).squeeze() 
        return prediction 
    
    def save_model(self, fname=DEFAULT_MODEL_PATH): 
        print(f"Saving model to path: {fname}") 
        torch.save(self.model.state_dict(), fname) 
    
    def load_model(self, fname=DEFAULT_MODEL_PATH):
        print(f"Loading model from path: {fname}") 
        self.model.load_state_dict(torch.load(fname, weights_only=True)) 
    
def test_inference(): 
    # Define parameters
    F = 5  # Number of features
    L = 3  # Number of loops
    C = 4  # Number of groupings (matrices)
    hidden_size = 10  # Hidden size for the encoder
    batch_size = 2  # Batch size (for testing with multiple batches)

    # Initialize the model
    input_size = L + 1  # Feature matrix has L+1 columns (features + grouping)
    model = BestMatrixSelector(input_size=input_size, hidden_size=hidden_size, C=C)

    # Create random input matrices (batch of C matrices per struct)
    random_matrices = []
    for _ in range(C):
        # Each matrix is of shape (batch_size, F, L+1)
        matrix = torch.randn(batch_size, F, L + 1)  # Random data simulating feature matrices
        random_matrices.append(matrix)

    # Forward pass through the model
    output = model(random_matrices)

    # Print the output to verify results
    print("Output predictions (one-hot encoding):", output)
    print("Shape of output:", output.shape)  # Should be (batch_size, C)

def test_training(): 
     # Define parameters
    F = 5  # Number of features
    L = 3  # Number of loops
    C = 4  # Number of groupings (matrices)
    hidden_size = 10  # Hidden size for the encoder
    batch_size = 2  # Batch size
    epochs = 1000  # Number of training epochs
    print(f"F: {F}, L: {L}, C: {C}")

    # Initialize the model
    input_size = L + 1  # Feature matrix has L+1 columns (features + grouping)
    model = BestMatrixSelector(input_size=input_size, hidden_size=hidden_size, C=C)
    encoder = MatrixEncoder(input_size=input_size, hidden_size=hidden_size) 
    

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs)):
        # Create random input matrices (batch of C matrices per struct)

        matrices = torch.randn(batch_size, C, F, L+1) 
        print("matrices shape: ", matrices.shape) 
        sums = torch.sum(matrices, dim=(2, 3)) 
        print("sums shape: ", sums.shape) 
        print("sums: ", sums)  
        targets = torch.argmax(sums, dim=1) 
        print("targets: ", targets)
        
        
        encoded = encoder(matrices) 
        print("encoded: ", encoded.shape) 

        # Forward pass through the model
        outputs = model(matrices)  # (batch_size, C)
        print("outputs shape: ", outputs.shape)
        print("outputs: ", outputs) 
        
        predictions = torch.argmax(outputs, dim=1) 
        print("predictions: ", predictions) 

        # Compute the loss
        loss = criterion(outputs, targets)
        print("loss: ", loss.item()) 

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    # Testing the trained model
    print("Training finished. Testing on random data...")
    with torch.no_grad(): 
        total_correct_cnt = 0; 
        total_test_cnt = 0 
        for i in range(10): 
            matrices = torch.randn(batch_size, C, F, L+1) 
            print("matrices shape: ", matrices.shape) 
            sums = torch.sum(matrices, dim=(2, 3)) 
            print("sums shape: ", sums.shape) 
            print("sums: ", sums)  
            targets = torch.argmax(sums, dim=1) 
            print("targets: ", targets)
            outputs = model(matrices) 
            predictions = torch.argmax(outputs, dim=1)  # Get the predicted matrix index
            print("Predicted best matrices:", predictions)
            correct = (targets == predictions)
            print(f"Test count: {correct.numel()} Correct count: {correct.sum()}") 
            total_correct_cnt += correct.sum() 
            total_test_cnt += int(correct.numel()) 
        total_correct_rate = (total_correct_cnt / total_test_cnt) * 100  
        print(f"Total test count: {total_test_cnt} Total correct count: {total_correct_cnt} Total correct rate: {total_correct_rate}%")
            
if __name__ == "__main__":
    test_training() 