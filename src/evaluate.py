# ---- src/evaluate.py ----
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from src.preprocess import load_data, preprocess_data
from src.model import AcupunctureModel
from sklearn.model_selection import train_test_split


def prepare_test_loader(df_encoded):
    """
    Prepare the test set DataLoader from the encoded DataFrame.

    Args:
        df_encoded (pd.DataFrame): Preprocessed DataFrame with features and target.

    Returns:
        DataLoader: DataLoader for the test set.
    """
    X = df_encoded.drop('acupoint', axis=1).values
    y = df_encoded['acupoint'].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)
    return test_loader, X_test_tensor.shape[1], len(set(y_test))


def evaluate_model(model, test_loader, criterion):
    """
    Evaluate a trained model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test DataLoader.
        criterion: Loss function.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


def main():
    df = load_data()
    df_encoded, _ = preprocess_data(df)
    test_loader, input_dim, output_dim = prepare_test_loader(df_encoded)

    model = AcupunctureModel(input_dim, output_dim)
    model.load_state_dict(torch.load('saved_models/acupuncture_model.pth'))

    criterion = nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, criterion)

if __name__ == '__main__':
    main()