import numpy as np
from pydantic import BaseModel
from pydantic_numpy.typing import NpNDArrayFp32 as point_precision_type



class LayerGradient(BaseModel):
    layer_index: str
    weight: point_precision_type
    bias: point_precision_type



point_precidion = np.float32



class DenseLayer:

    def __init__(self, input_size:int, number_of_neurons:int):
        low = -1
        high = 1
        self.weight = np.random.uniform(low, high, size=(input_size, number_of_neurons)).astype(point_precidion)
        self.bias = np.random.uniform(low, high, size=(1, number_of_neurons)).astype(point_precidion)

    
    def forward(self, input_matrix:np.ndarray):
        self.input_matrix = input_matrix.astype(point_precidion)
        output = np.dot(self.input_matrix, self.weight) + self.bias
        self.output = output

    
    def get_local_derivative_wrt_weights(self):
        self.gradient_wrt_weight_of_layer = self.input_matrix
        return self.gradient_wrt_weight_of_layer

    
    def get_local_derivative_wrt_inputs(self):
        self.gradient_wrt_output_of_previous_layer = self. weight
        return self.gradient_wrt_output_of_previous_layer



class DenseNetwork:

    def __init__(self, *denseLayers: list[DenseLayer]):
        self.layers = {f"layer{idx+1}": layer for idx, layer in enumerate(denseLayers)}

    def forward(self, input_matrix:np.ndarray):
        
        output = input_matrix

        for key, value in self.layers.items():
            value.forward(output)
            output = value.output

        self.logits = output
        return output

    
    def compute_loss(self, target:np.ndarray):
        error = self.logits - target.reshape(*self.logits.shape).astype(point_precidion)
        loss = error ** 2
        self.error = error
        self.loss = loss
        return loss

    
    def backward(self):
        gradients = []
        
        loss_gradient_wrt_prediction = (2 * self.error) / self.error.size
        cumulated_backpropagating_gradient = loss_gradient_wrt_prediction
        
        for idx, layer in list(self.layers.items())[::-1]:
            weight_gradient = np.dot(layer.get_local_derivative_wrt_weights().T, cumulated_backpropagating_gradient)
            bias_gradient = np.sum(cumulated_backpropagating_gradient, axis=0, keepdims=True)
            cumulated_backpropagating_gradient = np.dot(cumulated_backpropagating_gradient, layer.get_local_derivative_wrt_inputs().T)

            gradients.append(
                LayerGradient(
                    layer_index=idx,
                    weight=weight_gradient,
                    bias=bias_gradient
                )
            )

            self.gradients = [None] + gradients[::-1]


    def optimize(self, learning_rate:float):
        layers = list(zip([None]+list(self.layers.values()), self.gradients))
        
        for layer in layers[1:]:
            idx = layer[1].layer_index
            new_weight = layer[0].weight - (learning_rate * layer[1].weight)
            new_bias = layer[0].bias - (learning_rate * layer[1].bias)

            self.layers[idx].weight = new_weight
            self.layers[idx].bias = new_bias 