
import onnx
import numpy as np
from onnx.backend.test.case.node import expect

node = onnx.helper.make_node(
    'MatMul',
    inputs=['a', 'b'],
    outputs=['c'],
)

# 2d
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 3).astype(np.float32)
c = np.matmul(a, b)
expect(node, inputs=[a, b], outputs=[c],
       name='test_matmul_2d')

print(a.shape)
print(a)
print(b.shape)
print(b)
print(c.shape)
print(c)

# 3d
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 4, 3).astype(np.float32)
c = np.matmul(a, b)
expect(node, inputs=[a, b], outputs=[c],
       name='test_matmul_3d')

print(a.shape)
print(a)
print(b.shape)
print(b)
print(c.shape)
print(c)


# 4d
a = np.random.randn(1, 2, 3, 4).astype(np.float32)
b = np.random.randn(1, 2, 4, 3).astype(np.float32)
c = np.matmul(a, b)
expect(node, inputs=[a, b], outputs=[c],
       name='test_matmul_4d')

print(a.shape)
print(a)
print(b.shape)
print(b)
print(c.shape)
print(c)


