import tensorflow as tf


def charbonnier_loss(y_true: tf.Tensor, y_predicted: tf.Tensor, eps:float=1e-3) -> tf.Tensor:
    """
        Charbonnier Loss (Smooth L1-like robust loss)

        This is a differentiable variant of the L1 loss, often used in
        image restoration and optical flow because it is more robust to
        outliers than L2 and smoother than plain L1.

        Formula:
            L(x) = mean( sqrt( (y_true - y_predicted)^2 + eps^2 ) )

        Args:
            y_true (tf.Tensor):
                Ground truth tensor (e.g., image).
            y_predicted (tf.Tensor):
                Predicted tensor of the same shape as y_true.
            eps (float):
                Small constant added inside the square root to ensure
                numerical stability and smoothness near zero. Default = 1e-3.

        Returns:
            tf.Tensor:
                Scalar tensor representing the Charbonnier loss.
    """
    diff: tf.Tensor = y_true - y_predicted
    loss: tf.Tensor = tf.reduce_mean(tf.sqrt(tf.square(diff) + (eps * eps)))
    return loss

def edge_loss(y_true: tf.Tensor, y_predicted:tf.Tensor) -> tf.Tensor:
    """
       Edge-Aware Loss using Laplacian Pyramid Approximation

       This loss focuses on differences in image edges/structures rather than
       only raw pixel values. It approximates a Laplacian pyramid level by:
           1. Gaussian blurring (separable 5×5 kernel),
           2. Downsampling by factor 2,
           3. Upsampling back using sparse placement + Gaussian blur,
           4. Subtracting the reconstructed image from the original
              to obtain a band-pass / edge-emphasis representation.

       The Charbonnier loss is then computed between the Laplacian-filtered
       ground truth and prediction, encouraging sharper and more accurate edges.

       Args:
           y_true (tf.Tensor):
               Ground truth image batch, shape (B, H, W, C).
           y_predicted (tf.Tensor):
               Predicted image batch, same shape as y_true.

       Returns:
           tf.Tensor:
               Scalar tensor representing the edge-aware loss.
    """
    k: tf.Tensor = tf.constant([[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=tf.float32)
    kernel: tf.Tensor = tf.matmul(k, k, transpose_a=True)
    kernel: tf.Tensor = tf.expand_dims(kernel, axis=-1)
    kernel: tf.Tensor = tf.expand_dims(kernel, axis=-1)
    kernel: tf.Tensor = tf.repeat(kernel, repeats=3, axis=2)

    def conv_gauss(img):
        """
                Applies a 5x5 symmetric-padded Gaussian blur to the input image.

                This function uses depthwise convolution to apply the same Gaussian
                kernel independently to each channel.

                Steps:
                    • Symmetric pad by 2 pixels on all sides (to preserve image borders).
                    • Apply 5×5 depthwise Gaussian convolution.

                Args:
                    img (tf.Tensor):
                        Image batch tensor of shape (B, H, W, C).

                Returns:
                    tf.Tensor:
                        Gaussian-blurred image of shape (B, H, W, C).
        """
        if len(img.shape) == 3:
            img = tf.expand_dims(img, axis=0)
        img = tf.pad(img, [[0,0], [2,2], [2,2], [0,0]], mode='SYMMETRIC')
        return tf.nn.depthwise_conv2d(img, kernel, strides=[1,1,1,1], padding='VALID')

    def laplacian_kernel(current):
        """
                Computes a Laplacian pyramid level (edge band-pass component).

                This implementation approximates a classical Laplacian pyramid:

                    1. Smooth the image with Gaussian (low-pass)
                    2. Downsample by factor 2
                    3. Create an empty tensor and place downsampled pixels at even indices
                       → sparse upscaled grid
                    4. Smooth again to approximate an upsampled reconstruction
                    5. Subtract reconstructed image from original:
                            Laplacian = current - reconstructed

                The resulting tensor isolates high-frequency edges / details.

                Args:
                    current (tf.Tensor):
                        Input image batch (B, H, W, C).

                Returns:
                    tf.Tensor:
                        The Laplacian (edge band) of the input image.
        """
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
    loss: tf.Tensor = charbonnier_loss(laplacian_kernel(y_true), laplacian_kernel(y_predicted))
    return loss


def dummy_loss(y_true, y_predicted):
    """Dummy loss that returns 0 for unused outputs"""
    # y_predicted = y_predicted[1]
    return 0.0 * tf.reduce_mean(y_predicted)