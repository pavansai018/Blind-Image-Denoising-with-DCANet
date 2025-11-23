import tensorflow as tf


def charbonnier_loss(y_true, y_predicted, eps=1e-3):
    diff = y_true - y_predicted
    loss = tf.reduce_mean(tf.sqrt(tf.square(diff) + (eps * eps)))
    return loss

def edge_loss(y_true, y_predicted):
    k = tf.constant([[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=tf.float32)
    kernel = tf.matmul(k, k, transpose_a=True)
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.repeat(kernel, repeats=3, axis=2)

    def conv_gauss(img):
        img = tf.pad(img, [[0,0], [2,2], [2,2], [0,0]], mode='SYMMETRIC')
        return tf.nn.depthwise_conv2d(img, kernel, strides=[1,1,1,1], padding='VALID')

    def laplacian_kernel(current):
        filtered = conv_gauss(current)
        down = filtered[:, ::2, ::2, :]
        new_filter = tf.zeros_like(filtered)
        new_filter = tf.tensor_scatter_nd_update(
            new_filter,
            tf.stack(tf.meshgrid(tf.range(tf.shape(new_filter)[0]),
                                 tf.range(0, tf.shape(new_filter)[1], 2),
                                 tf.range(0, tf.shape(new_filter)[2], 2),
                                 tf.range(tf.shape(new_filter)[3]), indexing='ij'), axis=-1),
            down * 4
        )
        filtered = conv_gauss(new_filter)  # filter again
        diff = current - filtered
        return diff
    loss = charbonnier_loss(laplacian_kernel(y_true), laplacian_kernel(y_predicted))
    return loss
