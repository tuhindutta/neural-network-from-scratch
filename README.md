# Neural Network From Scratch

### [Repository Link](https://github.com/tuhindutta/neural-network-from-scratch)
### [Handwritten Notes](https://github.com/tuhindutta/neural-network-from-scratch/blob/main/notes.pdf)

A minimal neural network implementation built entirely with NumPy to demonstrate how a neural network learns through:

* Forward Propagation
* Mean Squared Error (MSE) Loss
* Backpropagation
* Gradient Descent

The repository was created as a learning project to expose the mechanics hidden behind deep learning frameworks such as TensorFlow and PyTorch.

Instead of calling:

```python
loss.backward()
optimizer.step()
```

this implementation explicitly computes gradients, propagates them through layers, and updates parameters manually.

---

## Project Goal

The objective of this repository is not to build a production-ready neural network framework.

The objective is to make the learning process visible.

By reading and modifying the code, you can observe:

* How predictions are generated
* How loss is computed
* How gradients flow backward through the network
* How weights and biases are updated
* How prediction error decreases over time

---

## Features

### Dense Neural Network

Implements a simple multi-layer dense network using NumPy.

### Manual Forward Propagation

Each layer performs matrix multiplication and bias addition without relying on external ML frameworks.

### Manual Backpropagation

Gradients are calculated explicitly using local derivatives and the chain rule.

### Gradient Descent Optimizer

Weights and biases are updated using vanilla gradient descent.

### Synthetic Regression Dataset

A mock housing-price dataset is generated to provide a predictable learning target.

### Type-Safe Gradient Objects

Uses Pydantic and pydantic-numpy for structured gradient storage.

### Jupyter Notebook Demo

A notebook version is included for interactive experimentation and quick result visualization.

---

## Repository Structure

```text
.
├── network.py
├── create_data.py
├── train.py
├── train.ipynb
├── requirements.txt
└── README.md
```

### network.py

Core implementation of the neural network.

Contains:

* DenseLayer
* DenseNetwork
* Forward propagation
* Loss computation
* Backpropagation
* Parameter optimization

### create_data.py

Generates a synthetic housing-price dataset and returns:

```python
X, y = get_data()
```

The generated dataset is normalized before training.

### train.py

Runs the complete training workflow:

1. Generate data
2. Build network
3. Train network
4. Print loss progression
5. Compare predictions against targets

### train.ipynb

Notebook version of the training script.

Useful for:

* Exploring outputs interactively
* Experimenting with architecture changes
* Visualizing training behavior

---

## Design Decisions

### Why No Activation Functions?

This project focuses on understanding forward propagation, backpropagation, and gradient descent.

Activation functions were intentionally omitted to keep the implementation small and make gradient flow easier to inspect.

### Why Regression Instead of Classification?

Regression allows training directly on network outputs using Mean Squared Error.

Adding classification would require introducing additional concepts such as:

* Softmax
* Cross Entropy Loss
* Probability Distributions

which would shift focus away from the learning mechanics being demonstrated.

### Why Multiple Layers?

Without activation functions, multiple dense layers collapse mathematically into a single linear transformation.

This means the additional layers do not increase model expressiveness.

They exist purely to demonstrate how gradients propagate through multiple stages during backpropagation.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
numpy==2.3.5
pandas==3.0.3
pydantic==2.12.5
pydantic-numpy==9.0.1
tabulate==0.10.0
```

---

## Running The Project

Execute:

```bash
python train.py
```

or open:

```text
train.ipynb
```

and run the notebook cells.

---

## Example Output

Before training, the network starts with randomly initialized weights and biases.

As a result, predictions differ significantly from the target values:

```text
Prediction: [-13.058812  -4.099347  -8.777883   8.434312  -8.366444
             10.810468   2.360078  -7.247364   4.142055  -9.279590]

Target    : [ -1.868641   1.264547   0.743604  -0.760140   1.011070
             -0.889023   1.120536  -0.984792   0.887069  -0.305459]
```

After repeated cycles of:

* Forward Propagation
* Loss Computation
* Backpropagation
* Gradient Descent

the model learns the underlying relationship in the dataset:

```text
Prediction: [-1.792489  1.281254  0.748594 -0.683408  1.082410
             -0.914453  1.145021 -0.909591  0.840811 -0.282333]

Target    : [-1.868641  1.264547  0.743604 -0.760140  1.011070
             -0.889023  1.120536 -0.984792  0.887069 -0.305459]
```

Training loss decreases consistently:

```text
+-------------+-------------+
| epoch       | loss        |
+=============+=============+
| 0           | 27.2907     |
| 20          | 0.334766    |
| 40          | 0.0104575   |
| 60          | 0.0037127   |
| 80          | 0.00354193  |
| 100         | 0.00353500  |
| 120         | 0.00353454  |
| 140         | 0.00353450  |
+-------------+-------------+
```

The reduction in prediction error and loss demonstrates that the network is successfully learning from data through gradient-based optimization.

---

## Accompanying Blog

This repository accompanies a detailed blog post that explains:

* Dense layers
* Forward propagation
* Mean Squared Error
* Local derivatives
* Chain rule
* Backpropagation
* Gradient descent

and maps the mathematics directly to the implementation contained in this repository.

The blog also includes handwritten derivations for readers who want to follow the underlying calculus step-by-step.

---

## License

This project is intended for educational purposes.

Feel free to explore, modify, extend, and experiment with the code.
