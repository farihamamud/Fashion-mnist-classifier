# Fashion-MNIST Classifier: PyTorch and NumPy

## Problem
Classify 28x28 grayscale pictures of clothing into 10 classes (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot).
I built the same small neural network twice: once with PyTorch, and once in plain NumPy, where I wrote the forward pass, backpropagation and training loop by hand.

## Data
Fashion-MNIST: 60,000 training images and 10,000 test images. The data downloads automatically through torchvision.

## Model
Flatten (784 inputs), then a layer of 128 units with ReLU, then 10 output scores.
Training: SGD, learning rate 0.1, batch size 64, 5 epochs, fixed seed 42.

## Results
| Version | Test accuracy after 5 epochs |
|---|---|
| PyTorch (`train.py`) | 0.843 |
| NumPy from scratch (`numpy_net.py`) | 0.840 |

Both come from one run, so differences of a point or two are not meaningful.

## What I found
Footwear and bags were easy. In the 5-epoch run, sandal, ankle boot and trouser all scored about 0.96-0.97.
Shirt was the hardest class (0.494). It was mostly mistaken for other tops: pullover (220), T-shirt (128) and coat (72), but never for a shoe.
This is probably because collars, buttons and sleeves are only a few pixels wide at 28x28. I did not test this explanation.

Training for 15 instead of 5 epochs raised overall test accuracy a little (0.843 to 0.855), but changed the per-class results a lot.
Shirt accuracy rose from 0.494 to 0.771, while sneaker fell from 0.840 to 0.707 and T-shirt from 0.800 to 0.715.
So overall accuracy hides what is happening inside the classes.

The NumPy network reached 0.840, close to PyTorch (0.843) with the same settings. This suggests my hand-written backpropagation is correct.

## Limitations
All numbers come from one run with seed 42. Before I fixed the seed, runs gave quite different per-class results (for example, pullover 0.73 in one run and 0.91 in another), so small differences between classes may be noise.
Test accuracy varied between epochs (0.872 at epoch 13, 0.855 at epoch 15) because the learning rate is large. I report the final epoch, not the best one.
I did not tune the model.

## How to run
pip install -r requirements.txt
python explore.py
python train.py
python numpy_net.py