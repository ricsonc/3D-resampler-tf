import tensorflow as tf
import ipdb
import numpy as np
import random
from tensorflow.python.framework import ops
from ipdb import set_trace as st

lib = tf.load_op_library('libinterp.so')

@ops.RegisterGradient("GridInterpolate3D")
def grid_interpolate3d_grad(op, grad):
    return lib.grid_interpolate3d_grad(
        grad, *op.inputs, soft_boundary = op.get_attr('soft_boundary')
    )


#differences
#1. soft / hard boundary (self explanatory)
#2. un(normalization) (self explanatory)

#these two should be good...
#3. grid is zyx, coordinates in zyx
# in original, grid is zyx, coords in zyx
#4. source is zyx
# in original, source is zyx as well

def tf_grid_sample(im, grid, S, use_native = False):

    gridshape = grid.shape
    
    num_batch = im.get_shape().as_list()[0]
    depth = im.get_shape().as_list()[1]
    height = im.get_shape().as_list()[2]
    width = im.get_shape().as_list()[3]
    channels = im.get_shape().as_list()[4]
    out_size = grid.get_shape().as_list()[1:-1]
    
    grid = tf.reshape(grid, [-1, 3])
    #x, y, z = tf.unstack(grid, axis = -1)
    z, y, x = tf.unstack(grid, axis = -1)
    
    depth_f = tf.to_float(depth)
    height_f = tf.to_float(height)
    width_f = tf.to_float(width)
    # Number of disparity interpolated.
    out_depth = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]
    zero = tf.zeros([], dtype='int32')
    # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
    max_z = tf.to_int32(tf.shape(im)[1] - 1)
    max_y = tf.to_int32(tf.shape(im)[2] - 1)
    max_x = tf.to_int32(tf.shape(im)[3] - 1)

    # Converts scale indices from [-1, 1] to [0, width/height/depth].
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0
    z = (z + 1.0) * (depth_f - 1.0) / 2.0

    grid = tf.stack([z,y,x], axis = -1)
    grid = tf.reshape(grid, gridshape)

    if use_native:
        return lib.grid_interpolate3d(im, grid, soft_boundary = True)
    else:
        raw_out = tf_grid_sample_unnormalized(im, grid)
        return tf.reshape(raw_out, im.shape)

def tf_grid_sample_unnormalized(im, grid):
    #rename some variables, do some reshaping
    
    out_size = grid.get_shape().as_list()[1:-1]    
    grid = tf.reshape(grid, [-1, 3])
    z, y, x = tf.unstack(grid, axis = -1)
    BS = int(im.shape[0])

    #################
    
    num_batch = im.get_shape().as_list()[0]
    depth = im.get_shape().as_list()[1]
    height = im.get_shape().as_list()[2]
    width = im.get_shape().as_list()[3]
    channels = im.get_shape().as_list()[4]

    x = tf.to_float(x)
    y = tf.to_float(y)
    z = tf.to_float(z)
    
    depth_f = tf.to_float(depth)
    height_f = tf.to_float(height)
    width_f = tf.to_float(width)
    
    # Number of disparity interpolated.o
    out_depth = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]
    
    zero = tf.zeros([], dtype='int32')
    # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
    max_z = tf.to_int32(tf.shape(im)[1] - 1)
    max_y = tf.to_int32(tf.shape(im)[2] - 1)
    max_x = tf.to_int32(tf.shape(im)[3] - 1)

    #normalization already done

    x0 = tf.to_int32(tf.floor(x))
    x1 = x0 + 1
    y0 = tf.to_int32(tf.floor(y))
    y1 = y0 + 1
    z0 = tf.to_int32(tf.floor(z))
    z1 = z0 + 1

    x0_clip = tf.clip_by_value(x0, zero, max_x)
    x1_clip = tf.clip_by_value(x1, zero, max_x)
    y0_clip = tf.clip_by_value(y0, zero, max_y)
    y1_clip = tf.clip_by_value(y1, zero, max_y)
    z0_clip = tf.clip_by_value(z0, zero, max_z)
    z1_clip = tf.clip_by_value(z1, zero, max_z)
    
    dim3 = width
    dim2 = width * height
    dim1 = width * height * depth

    base = tf.constant(
        np.concatenate([np.array([i*dim1] * out_depth * out_height * out_width)
                        for i in range(BS)]).astype(np.int32)
    )

    base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
    base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
    base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
    base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

    idx_z0_y0_x0 = base_z0_y0 + x0_clip
    idx_z0_y0_x1 = base_z0_y0 + x1_clip
    idx_z0_y1_x0 = base_z0_y1 + x0_clip
    idx_z0_y1_x1 = base_z0_y1 + x1_clip
    idx_z1_y0_x0 = base_z1_y0 + x0_clip
    idx_z1_y0_x1 = base_z1_y0 + x1_clip
    idx_z1_y1_x0 = base_z1_y1 + x0_clip
    idx_z1_y1_x1 = base_z1_y1 + x1_clip

    # Use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.to_float(im_flat)
    i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
    i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
    i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
    i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
    i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
    i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
    i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
    i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)

    # Finally calculate interpolated values.
    x0_f = tf.to_float(x0)
    x1_f = tf.to_float(x1)
    y0_f = tf.to_float(y0)
    y1_f = tf.to_float(y1)
    z0_f = tf.to_float(z0)
    z1_f = tf.to_float(z1)
    
    # Check the out-of-boundary case.
    x0_valid = tf.to_float(
        tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
    x1_valid = tf.to_float(
        tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
    y0_valid = tf.to_float(
        tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
    y1_valid = tf.to_float(
        tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
    z0_valid = tf.to_float(
        tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
    z1_valid = tf.to_float(
        tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))

    if True: #out of range mode "boundary"
        x0_valid = tf.ones_like(x0_valid)
        x1_valid = tf.ones_like(x1_valid)
        y0_valid = tf.ones_like(y0_valid)
        y1_valid = tf.ones_like(y1_valid)
        z0_valid = tf.ones_like(z0_valid)
        z1_valid = tf.ones_like(z1_valid)

    w_z0_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                 (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                1)
    w_z0_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                 (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                1)
    w_z0_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                 (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                1)
    w_z0_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                 (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                1)
    w_z1_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                 (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                1)
    w_z1_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                 (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                1)
    w_z1_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                 (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                1)
    w_z1_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                 (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                1)

    weights_summed = (
        w_z0_y0_x0 +
        w_z0_y0_x1 +
        w_z0_y1_x0 +
        w_z0_y1_x1 +
        w_z1_y0_x0 +
        w_z1_y0_x1 +
        w_z1_y1_x0 +
        w_z1_y1_x1
    )

    output = tf.add_n([
        w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1,
        w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1,
        w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1,
        w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1
    ])

    return output

def rand_trans_single():
    if True:
        theta, phi, gamma = random.random() * 2 * np.pi, random.random() * 2 * np.pi, random.random() * 2 * np.pi
    else:
        print('no rotation at all!')
        theta, phi, gamma = 0.0, 0.0, 0.0
    
    ct = np.cos(theta)
    cp = np.cos(phi)
    cg = np.cos(gamma)
    st_ = np.sin(theta)
    sp = np.sin(phi)
    sg = np.sin(gamma)

    rx = np.array([
        [1, 0, 0],
        [0, ct, -st_],
        [0, st_, ct],
    ])

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    rz = np.array([
        [cg, -sg, 0],
        [sg, cg, 0],
        [0, 0, 1],
    ])

    combined = np.matmul(rx, np.matmul(ry, rz))

    return combined

def rand_trans(B):
    return np.stack([rand_trans_single() for _ in range(B)], axis = 0).astype(np.float32)

def numpy_grid(S):
    sample_axis = np.linspace(-1, 1, S)
    #in default "xy" mode, this returns really strange results!
    return np.stack(np.meshgrid(sample_axis, sample_axis, sample_axis, indexing="ij"), axis = -1).astype(np.float32)
