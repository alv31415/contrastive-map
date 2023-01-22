import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Class encoding a simple multi-layer perceptron, used for encoding and prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation = nn.ReLU(), use_bias = True, use_batch_norm = True):
        """
        input_dim: an int, indicating the input dimension for the MLP.
        hidden_dim: an int, indicating the hidden dimension for the MLP.
        output_dim: an int, indicating the output dimension for the MLP.
        activation: a Pytorch activation (from the nn module), providing the non-linearity for the MLP.
        use_bias: a boolean. If true, the linear layers will use a bias weight.
        use_batch_norm: a boolean. If true, applies batch normalisation after the hidden layer.
        """
        super(MLP, self).__init__()
        
        # network layers for the projection head
        self.lin_hidden = nn.Linear(in_features = input_dim, 
                                    out_features = hidden_dim, 
                                    bias = use_bias)
        self.batch_norm = nn.BatchNorm1d(num_features = hidden_dim) if use_batch_norm else nn.Identity()
        self.activation = activation
        self.lin_output = nn.Linear(in_features = hidden_dim, 
                                    out_features = output_dim, 
                                    bias = use_bias)
        
        # define the model
        self.mlp = nn.Sequential(
                self.lin_hidden,
                self.batch_norm,
                self.activation,
                self.lin_output
                )
        
    def forward(self, x):
        """
        A forward pass through the MLP.
        """
        return self.mlp(x)