import numpy as np
from torchvision import datasets

train = datasets.FashionMNIST("data", train=True, download=True)
test = datasets.FashionMNIST("data", train=False, download=True)

X_train = train.data.numpy().reshape(-1, 784) / 255.0
y_train = train.targets.numpy()
X_test = test.data.numpy().reshape(-1, 784) / 255.0
y_test = test.targets.numpy()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train.min(), X_train.max())

rng = np.random.default_rng(42)

W1 = rng.normal(0, 0.05, (784, 128))
b1 = np.zeros(128)
W2 = rng.normal(0, 0.05, (128, 10))
b2 = np.zeros(10)

def forward(X):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)        # ReLU
    z2 = a1 @ W2 + b2             # 10 scores per picture
    return z1, a1, z2

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y):
    return -np.log(probs[np.arange(len(y)), y]).mean()

X_batch, y_batch = X_train[:64], y_train[:64]
z1, a1, z2 = forward(X_batch)
probs = softmax(z2)

print(z1.shape, a1.shape, z2.shape)
print(probs[0].round(3), probs[0].sum())
print("loss:", cross_entropy(probs, y_batch))
print("accuracy:", (probs.argmax(axis=1) == y_batch).mean())

def backward(X, y, z1, a1, probs):
    n = len(y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1

    dz2 = (probs - onehot) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2

lr = 0.1
for step in range(101):
    z1, a1, z2 = forward(X_batch)
    probs = softmax(z2)
    if step % 20 == 0:
        print(step, "loss:", cross_entropy(probs, y_batch))
    dW1, db1, dW2, db2 = backward(X_batch, y_batch, z1, a1, probs)
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# fresh weights, the demo loop above trained them on only 64 pictures
W1 = rng.normal(0, 0.05, (784, 128))
b1 = np.zeros(128)
W2 = rng.normal(0, 0.05, (128, 10))
b2 = np.zeros(10)

def accuracy(X, y):
    _, _, z2 = forward(X)
    return (z2.argmax(axis=1) == y).mean()

print("Accuracy before training:", accuracy(X_test, y_test))

for epoch in range(5):
    order = rng.permutation(len(X_train))          # shuffle
    for i in range(0, len(order), 64):
        idx = order[i:i + 64]
        Xb, yb = X_train[idx], y_train[idx]
        z1, a1, z2 = forward(Xb)
        probs = softmax(z2)
        dW1, db1, dW2, db2 = backward(Xb, yb, z1, a1, probs)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    print(f"Epoch {epoch + 1}: test accuracy {accuracy(X_test, y_test):.3f}")