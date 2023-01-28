import torch.nn as nn

from hook import Hook

class EncoderProjectorNN(nn.Module):
    """
    Class for a network which encodes an input and projects it into a latent space.
    """
    def __init__(self, encoder, projector, encoder_layer_idx = -1):
        """
        encoder: a nn.Module, containing an encoder network.
        projector: a nn.Module, containing a projector network.
        encoder_layer_idx: an int, corresponding to the index of the layer of the encoder 
                           which is actually used for encoding.
                           For instance, in BYOL, if using a ResNet as an encoder, 
                           the paper uses the output of the last average pooliing layer, 
                           which is the penultimate layer of the ResNet 
                           (corresponding to encoder_layer_idx = -2).
        """
        super(EncoderProjectorNN, self).__init__()
        
        self.encoder = encoder
        self.projector = projector
        self.encoder_layer_idx = encoder_layer_idx
        
        # if the encoding layer isn't the last one, add a hook to save the output of the layer
        if self.encoder_layer_idx != -1:
            self.hook = Hook()
            layers = [*self.encoder.children()]
            encoder_layer = layers[self.encoder_layer_idx]
            self.hook.set_hook(encoder_layer)
            
    def encode(self, x):
        """
        Encodes an input x, according to the encoder network and self.encoder_layer_idx.
        """
        
        encoded = self.encoder(x)
        
        if self.encoder_layer_idx != -1:
            # need to fix the output to remove the additional dimensions
            # for instance, images sent as 4-dimensional input (N,C,W,H) will remain 4 dimensional 
            # after passing through the encoder, and rely on final layers to reduce this to
            # 2 dimensions (N,X). 
            # Thus, if self.encoder_layer_idx != -1, we are missing this dimensionality reduction step, 
            # so we need to compensate for it.
            encoded = self.hook.output.reshape(encoded.shape[0],-1)
            
        return encoded
    
    def project(self, x):
        """
        Projects an input x into latent space, according to the projector network.
        """
        return self.projector(x)
    
    def forward(self, x):
        """
        A forward pass through the model, involving first encoding and then projecting.
        """
        encode_x = self.encode(x)
        return self.project(encode_x)