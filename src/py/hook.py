# based on the Hook class by Kennethborup:
# https://github.com/Kennethborup/BYOL
class Hook():
    """
    A simple hook class that returns the output of a layer of a model during forward pass.
    """
    def __init__(self):
        self.output = None
        
    def set_hook(self, module):
        """
        Attaches hook to model.
        """
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, _, output):
        """
        Saves the wanted information.
        """
        self.output = output