import onnx
import numpy as np
from onnx.backend.test.case.node import expect

node = onnx.helper.make_node(
    'Where',
    inputs=['condition', 'x', 'y'],
    outputs=['z'],
)

condition = np.array([[1, 0], [1, 1]], dtype=bool)
x = np.array([[1, 2], [3, 4]], dtype=np.float32)
y = np.array([[9, 8], [7, 6]], dtype=np.float32)
z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
expect(node, inputs=[condition, x, y], outputs=[z],
       name='test_where_example')


print(condition.shape)
print(condition)

print(x.shape)
print(x)

print(y.shape)
print(y)

print(z.shape)
print(z)
