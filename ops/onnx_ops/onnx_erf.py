
import onnx
import numpy as np
from onnx.backend.test.case.node import expect
import math

node = onnx.helper.make_node(
    'Erf',
    inputs=['x'],
    outputs=['y'],
)

x = np.random.randn(1, 3, 32, 32).astype(np.float32)
y = np.vectorize(math.erf)(x).astype(np.float32)

print(x.shape)
print(x)

print(y.shape)
print(y)

node = onnx.helper.make_node(
    'Tanh',
    inputs=['x'],
    outputs=['y'],
)

y = np.tanh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_tanh')

print(y.shape)
print(y)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.tanh(x)  # expected output [-0.76159418, 0., 0.76159418]
expect(node, inputs=[x], outputs=[y],
       name='test_tanh_example')

print(x.shape)
print(x)

print(y.shape)
print(y)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.tanh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_tanh')

print(x.shape)
print(x)

print(y.shape)
print(y)
