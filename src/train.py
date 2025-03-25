# ---- src/train.py ----
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from src.preprocess import load_data, preprocess_data
from src.model import AcupunctureModel

from sklearn.model_selection import train_test_split


def prepare_dataloaders(df_encoded):
    """
    Split the dataset and prepare PyTorch dataloaders.

    Args:
        df_encoded (pd.DataFrame): Encoded feature DataFrame.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders.
    """
    X = df_encoded.drop('acupoint', axis=1).values
    y = df_encoded['acupoint'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

    return train_loader, test_loader, X_train_tensor.shape[1], len(set(y))


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the neural network.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Dataloader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs (int): Number of training epochs.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")


def save_model(model, path='saved_models/acupuncture_model.pth'):
    """
    Save the trained model to disk.

    Args:
        model (nn.Module): Trained model.
        path (str): Path to save the model file.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    df = load_data()
    df_encoded, _ = preprocess_data(df)
    train_loader, _, input_dim, output_dim = prepare_dataloaders(df_encoded)

    model = AcupunctureModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer)
    save_model(model)

if __name__ == '__main__':
    main()