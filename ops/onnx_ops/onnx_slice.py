import onnx
import numpy as np
from onnx.backend.test.case.node import expect

node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes', 'steps'],
    outputs=['y'],
)

pass

x = np.random.randn(6, 4, 5).astype(np.float32)
y = x[0:3, 0:4]

starts = np.array([0, 0], dtype=np.int64)
ends = np.array([3, 4], dtype=np.int64)
axes = np.array([0, 1], dtype=np.int64)
steps = np.array([1, 1], dtype=np.int64)

print(x.shape)
print(x)

print(y.shape)
print(y)

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_slice')

input = np.random.randn(5, 6, 7).astype(np.float32)
for x in range(5):
  for y in range(6):
    for z in range(7):
      input[x,y,z] = 6*7*x + 7*y +z
print(input)

result = input[1:4, 2:5, 3:6]
print(result)
