
import numpy as np
import tensorflow as tf

"""
Code forked from tensorlayer.
"""

# __all__ = [
#     "Layer",
#     "DenseLayer",
# ]

# set_keep = locals()
set_keep = globals()
set_keep['_layers_name_list'] = []
set_keep['name_reuse'] = False

logging = tf.logging

class Layer(object):
    """
    The basic :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    inputs : :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : str or None
        A unique layer name.

    Methods
    ---------
    print_params(details=True, session=None)
        Print all parameters of this network.
    print_layers()
        Print all outputs of all layers of this network.
    count_params()
        Return the number of parameters of this network.
    """

    def __init__(self, inputs=None, name='layer'):
        self.inputs = inputs
        scope_name = tf.get_variable_scope().name
        if scope_name:
            name = scope_name + '/' + name
        if (name in set_keep['_layers_name_list']) and set_keep['name_reuse'] == False:
            raise Exception("Layer '%s' already exists, please choice other 'name' or reuse this layer\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)\
            \nAdditional Informations: http://tensorlayer.readthedocs.io/en/latest/modules/layers.html?highlight=clear_layers_name#tensorlayer.layers.clear_layers_name"
                            % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)

    def print_params(self, details=True, session=None):
        """Print all info of parameters in the network"""
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    # logging.info("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                    val = p.eval(session=session)
                    logging.info("  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".format(
                        i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std()))
                except Exception as e:
                    logging.info(str(e))
                    raise Exception("Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")
            else:
                logging.info("  param {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        logging.info("  num of params: %d" % self.count_params())

    def print_layers(self):
        """Print all info of layers in the network"""
        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            logging.info("  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name))

    def count_params(self):
        """Return the number of parameters in the network"""
        n_params = 0
        for _i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except Exception:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

    def __str__(self):
        # logging.info("\nIt is a Layer class")
        # self.print_params(False)
        # self.print_layers()
        return "  Last layer is: %s" % self.__class__.__name

class InputLayer(Layer):
    """
    The :class:`InputLayer` class is the starting layer of a neural network.
    Parameters
    ----------
    inputs : placeholder or tensor
        The input of a network.
    name : str
        A unique layer name.
    """

    def __init__(self, inputs=None, name='input'):
        Layer.__init__(self, inputs=inputs, name=name)
        logging.info("InputLayer  %s: %s" % (self.name, inputs.get_shape()))
        self.outputs = inputs
        self.all_layers = []
        self.all_params = []
        self.all_drop = {}
