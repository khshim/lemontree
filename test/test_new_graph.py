import theano
import theano.tensor as T

from lemontree.layers.dense import DenseLayer
from lemontree.layers.activation import ReLU, Sigmoid

from lemontree.graphs.graph import SimpleGraph
from lemontree.utils.param_utils import print_tags_in_params

graph = SimpleGraph('test', 100)

graph.add_layer(DenseLayer((10,),(10,)), get_from=[])
graph.add_layer(DenseLayer((10,),(10,)), get_from=[-1])
graph.add_layer(DenseLayer((10,),(10,)), get_from=[-1])
graph.add_layer(DenseLayer((10,),(10,)), get_from=[1])
graph.add_layer(DenseLayer((10,),(10,)), get_from=[-1])
graph.add_layer(DenseLayer((10,),(10,)), get_from=[-3])
graph.add_layer(DenseLayer((10,),(10,)), get_from=[-1])
graph.add_layer(DenseLayer((10,),(10,)), get_from=[-3])

print(graph.connections)

input_ = T.fmatrix('X')

out_ = graph.get_output([input_], -1, 1)

param_ = graph.get_params()
print_tags_in_params(param_)