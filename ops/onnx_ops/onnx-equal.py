import onnx
import numpy as np
from onnx.backend.test.case.node import expect

node = onnx.helper.make_node(
    'Equal',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
y = (np.random.randn(4,5) * 10).astype(np.int32)
z = np.equal(x, y)

print(x.shape)
print(x)

print(y.shape)
print(y)

print(z.shape)
print(z)
