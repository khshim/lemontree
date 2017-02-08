import theano
import theano.tensor as T

from lemontree.layers.dense import DenseLayer
from lemontree.layers.activation import ReLU, Sigmoid
from lemontree.layers.objectives import CategoricalCrossentropy

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
graph.add_layer(CategoricalCrossentropy(), get_from=[-1])

print(graph.connections)

input_ = T.fmatrix('X')
label_ = T.ivector('L')

out_, list_ = graph.get_output({-8: [input_], -1:[label_]}, -1, -8)

param_ = graph.get_params(list_)
print_tags_in_params(param_)
print(out_.ndim)