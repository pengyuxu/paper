''' 
/* SHREC 2025
Marco Guerra

*/
'''

# NN models

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        
        super(SimpleNN, self).__init__()

        self.dr = nn.Dropout(p=.25) # dropout layer
        self.fc1 = nn.Linear(input_dim, 111) 
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(111, 100) 
        self.fc3 = nn.Linear(100,97) 
        self.fc4 = nn.Linear(97, num_classes)  # Output layer
        

    def forward(self, x):
        #x = self.dr(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dr(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dr(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dr(x)
        x = self.fc4(x)  # No softmax, since CrossEntropyLoss applies it internally
        return x

class Autoencoder(nn.Module):
    def __init__(self, size : int = 75 , noise : float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 15)  # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 50),
            nn.ReLU(),
            nn.Linear(50, size)
        )

        self.noise = noise

    def forward(self, x):
        x = x + self.noise * torch.randn_like(x)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self,x):

        return self.encoder(x)



def train(N_epochs,model,criterion, optimizer, train_loader, val_loader, save_path, patience=25 ):

    best_loss = float('inf')
    counter = 0

    train_losses = []
    #true_tr_losses = []
    val_losses = []
    acc_epochs = []
    accuracies = []

    for epoch in range(N_epochs):  # Train for up to N_epochs epochs

        running_train_loss = 0.0
        
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Loss
        
        model.eval()

        
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

        if epoch % 20 == 0: # check accuracy

            model.eval()  # Set model to evaluation mode
            batch_acc = []
            acc_epochs.append(epoch)
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    predicted = torch.argmax(outputs, dim=1)  # Get class with highest probability
                    predicted = predicted.cpu().numpy()
                    batch_acc.append( accuracy_score(predicted, y_batch ) )
                    
            
            batch_acc = np.array(batch_acc)
            accuracies.append( np.mean(batch_acc) )
    
        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0  # Reset counter if validation loss improves
            torch.save(model.state_dict(), save_path)  # Save best model
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break  # Stop training

        # Live plot update
        clear_output(wait=True)

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        #plt.plot(true_tr_losses, label="True Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim([ 0.0 , 2.5 ])
        plt.grid()
        plt.title("Training vs Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(acc_epochs, accuracies, 'o-', label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy ')
        plt.ylim([ 0.0 , 1.0 ])
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()


    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")


def predict(model, val_loader, device):

    model.eval()  # Set model to evaluation mode
    y_pred = []
    
    with torch.no_grad():  # No gradient calculation needed
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)  # Get class with highest probability
            y_pred.extend(predicted.cpu().numpy())

    return y_pred




