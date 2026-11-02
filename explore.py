from torchvision import datasets, transforms
import matplotlib.pyplot as plt

train = datasets.FashionMNIST("data", train=True, download=True, transform=transforms.ToTensor())
print(len(train))
image, label = train[0]
print(image.shape, label, train.classes[label])
plt.imshow(image.squeeze(), cmap="gray")
plt.title(train.classes[label])
plt.show()