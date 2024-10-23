import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        self.params = {}

        # Initialize weights and biases for the first layer (input -> hidden)
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)

        # Initialize weights and biases for the second layer (hidden -> output)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def parameters(self):
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #
        return self.params

        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return params

    def forward(self, X):
        scores, cache = None, None
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        # Unpack parameters
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # First layer: FC layer + ReLU
        fc1_out, fc1_cache = fc_forward(X, W1, b1)  # FC layer
        relu_out, relu_cache = relu_forward(fc1_out)  # ReLU activation

        # Second layer: FC layer (no activation, just raw scores)
        scores, fc2_cache = fc_forward(relu_out, W2, b2)

        # Cache everything needed for the backward pass
        cache = (fc1_cache, relu_cache, fc2_cache)
        
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
            # Unpack cache from the forward pass
        fc1_cache, relu_cache, fc2_cache = cache

        # Backprop through the second fully connected layer (output layer)
        grad_relu_out, grad_W2, grad_b2 = fc_backward(grad_scores, fc2_cache)

        # Backprop through the ReLU activation
        grad_fc1_out = relu_backward(grad_relu_out, relu_cache)

        # Backprop through the first fully connected layer
        grad_X, grad_W1, grad_b1 = fc_backward(grad_fc1_out, fc1_cache)

        # Store gradients in a dictionary
        grads = {
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2
        }
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return grads