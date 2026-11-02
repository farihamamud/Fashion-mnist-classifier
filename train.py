import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)

train_data = datasets.FashionMNIST("data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST("data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
print(model)
print(model(images).shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def accuracy(loader):
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)

print("Accuracy before training:", accuracy(test_loader))

for epoch in range(15):
    for images, labels in train_loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: last batch loss {loss.item():.3f}, test accuracy {accuracy(test_loader):.3f}")


classes = train_data.classes
correct = torch.zeros(10)
total = torch.zeros(10)
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        for label, pred in zip(labels, preds):
            total[label] += 1
            correct[label] += int(label == pred)

for i, name in enumerate(classes):
    print(f"{name}: {correct[i] / total[i]:.3f}")

confusion = torch.zeros(10, 10, dtype=torch.int)
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        for label, pred in zip(labels, preds):
            confusion[label, pred] += 1

shirt = classes.index("Shirt")
for j, name in enumerate(classes):
    print(f"Shirt predicted as {name}: {confusion[shirt, j].item()}")        