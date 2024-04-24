import torch
from torch.utils.data import DataLoader

def test(model, test_data):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_data_loader=DataLoader(
            test_data,
            model.bs,
            shuffle=True
        )

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_data_loader:

            data, target = data.to(device), target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy
