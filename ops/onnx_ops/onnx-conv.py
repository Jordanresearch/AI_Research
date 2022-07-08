import onnx
import numpy as np
from onnx.backend.test.case.node import expect

x = np.random.randn(3,244,244).astype(np.float32)
W = np.random.randn(768,3,16,16).astype(np.float32)
y = np.random.randn(14,14).astype(np.float32)

print(x.shape)
#print(x)
print(W.shape)
#print(W)
print(y.shape)
#print(y)

# Convolution with padding
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[16, 16],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[0, 0, 0, 0],
)
y_with_padding = np.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                             [33., 54., 63., 72., 51.],
                             [63., 99., 108., 117., 81.],
                             [93., 144., 153., 162., 111.],
                             [72., 111., 117., 123., 84.]]]]).astype(np.float32)

# Convolution without padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[16, 16],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[0, 0, 0, 0],
    strides=[16, 16]
)

print(x.shape)
#print(x)
print(W.shape)
#print(W)
print(y.shape)
#print(y)

y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                [99., 108., 117.],
                                [144., 153., 162.]]]]).astype(np.float32)

x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                [5., 6., 7., 8., 9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.],
                [20., 21., 22., 23., 24.]]]]).astype(np.float32)
W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

# Convolution with auto_pad='SAME_LOWER' and strides=2
node = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    auto_pad='SAME_LOWER',
    kernel_shape=[3, 3],
    strides=[2, 2],
)
y = np.array([[[[12., 27., 24.],
             [63., 108., 81.],
             [72., 117., 84.]]]]).astype(np.float32)
expect(node, inputs=[x, W], outputs=[y],
       name='test_conv_with_autopad_same')
