import onnx
import numpy as np
from onnx.backend.test.case.node import expect

node = onnx.helper.make_node(
    'Squeeze',
    inputs=['x', 'axes'],
    outputs=['y'],
)
x = np.random.randn(64, 64, 1, 1).astype(np.float32)
axes = np.array([2], dtype=np.int64)
y = np.squeeze(x, axis=2)
print(x.shape)
print(y.shape)

expect(node, inputs=[x, axes], outputs=[y],
       name='test_squeeze')
