import onnx
import numpy as np
from onnx.backend.test.case.node import expect

node = onnx.helper.make_node(
    'Expand',
    inputs=['data', 'new_shape'],
    outputs=['expanded'],
)
shape = [3, 1]
data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[1.], [2.], [3.]]
new_shape = [2, 1, 6]
expanded = data * np.ones(new_shape, dtype=np.float32)
#print(expanded)
#[[[1., 1., 1., 1., 1., 1.],
#  [2., 2., 2., 2., 2., 2.],
#  [3., 3., 3., 3., 3., 3.]],
#
# [[1., 1., 1., 1., 1., 1.],
#  [2., 2., 2., 2., 2., 2.],
#  [3., 3., 3., 3., 3., 3.]]]
new_shape = np.array(new_shape, dtype=np.int64)
expect(node, inputs=[data, new_shape], outputs=[expanded],
       name='test_expand_dim_changed')

print(data.shape)
print(data)

print(expanded.shape)
print(expanded)

node = onnx.helper.make_node(
    'Expand',
    inputs=['data', 'new_shape'],
    outputs=['expanded'],
)
shape = [3, 1]
new_shape = [3, 4]
data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[1.], [2.], [3.]]
expanded = np.tile(data, 4)
#print(expanded)
#[[1., 1., 1., 1.],
# [2., 2., 2., 2.],
# [3., 3., 3., 3.]]
new_shape = np.array(new_shape, dtype=np.int64)
expect(node, inputs=[data, new_shape], outputs=[expanded],
       name='test_expand_dim_unchanged')

print(data.shape)
print(data)

print(expanded.shape)
print(expanded)
