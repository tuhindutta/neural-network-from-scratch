import numpy as np
import pandas as pd
from network import DenseLayer, DenseNetwork
from create_data import get_data


X, y = get_data(n_samples=10000)

l1 = DenseLayer(4, 10)
l2 = DenseLayer(10, 25)
l3 = DenseLayer(25, 1)

network = DenseNetwork(
    l1,
    l2,
    l3
)

number_of_idx_to_print = 10

network.forward(X)
predictions = network.logits.flatten()
with np.printoptions(precision=6, suppress=True, floatmode='fixed', linewidth=200):
    print("Prediction:", predictions[:number_of_idx_to_print])
    print("Target    :", y[:number_of_idx_to_print])
    print("Error     :", y[:number_of_idx_to_print] - predictions[:number_of_idx_to_print])
    print()

epoch_res = pd.DataFrame(columns=["loss"])
epoch_res.index.name = "epoch"

for epoch in range(150):

    network.forward(X)

    network.compute_loss(y)

    loss = np.mean(network.loss)

    network.backward()

    network.optimize(0.001)

    if epoch % 20 == 0:
        epoch_res.loc[epoch] = loss


network.forward(X)

predictions = network.logits.flatten()
epoch_res.loc["final epoch"] = loss

print(epoch_res.to_markdown(tablefmt="grid")+"\n")

with np.printoptions(precision=6, suppress=True, floatmode='fixed', linewidth=200):
    print("Prediction:", predictions[:number_of_idx_to_print])
    print("Target    :", y[:number_of_idx_to_print])
    print("Error     :", y[:number_of_idx_to_print] - predictions[:number_of_idx_to_print])