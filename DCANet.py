import tensorflow as tf

def noise_estimation_network(input_layer:tf.keras.layers.Layer=tf.keras.layers.Input(shape=(128, 128, 3)),
                             filters:int = 64, kernel_size:int = 3, out_channels: int = 3) -> tf.keras.layers.Layer:
    """
        Builds a convolutional neural network for estimating image noise.

        This network takes an RGB image tensor (default: 128×128×3) and predicts a
        noise map of the same spatial resolution. It uses a series of convolutional
        blocks with ReLU activations and Batch Normalization to extract noise-related
        features. The final output is generated using a `tanh` activation, making the
        network suitable for tasks where the noise values are normalized in the range [-1, 1].

        Args:
            input_layer (tf.keras.layers.Layer):
                Input tensor or Keras Input layer. Default is an image-shaped input.
            filters (int):
                Number of convolutional filters for intermediate layers. Default = 64.
            kernel_size (int):
                Size of the convolution kernel. Default = 3.
            out_channels (int):
                Number of channels in the output noise map. Default = 3.

        Returns:
            tf.keras.layers.Layer:
                Output tensor representing the estimated noise map.
    """
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',)(input_layer)
    x = tf.keras.layers.ReLU()(x)
    for i in range(5):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding='same', )(x)
    outputs = tf.keras.layers.Activation('tanh')(x)

    return outputs

def spatial_attention_module(inputs: tf.keras.layers.Layer, filters:int=64, kernel_size:int=5) -> tf.keras.layers.Layer:
    """
        Spatial Attention Module (SAM)

        This module generates a spatial attention map by aggregating spatial
        information using both average pooling (GAP) and max pooling (GMP)
        across the channel dimension.
        The concatenated descriptor is passed through a convolution to
        generate a single-channel spatial attention mask (sigmoid-activated),
        which highlights the important spatial regions in the input feature map.

        Args:
            inputs (tf.keras.layers.Layer):
                Input feature map of shape (H, W, C).
            filters (int):
                Unused (kept for interface consistency). Default = 64.
            kernel_size (int):
                Kernel size for spatial convolution (default = 5).

        Returns:
            tf.keras.layers.Layer:
                Output tensor after applying spatial attention, computed as:
                `inputs * spatial_mask`
    """
    gap: tf.keras.layers.Layer = tf.keras.layers.Lambda(lambda y: tf.reduce_mean(y, axis=-1, keepdims=True))(inputs)
    gmp: tf.keras.layers.Layer = tf.keras.layers.Lambda(lambda y: tf.reduce_max(y, axis=-1, keepdims=True))(inputs)
    spatial_pool: tf.keras.layers.Layer = tf.keras.layers.Concatenate(axis=-1)([gap, gmp])
    conv2d_1: tf.keras.layers.Layer = tf.keras.layers.Conv2D(
        filters=1,  # 1 output channel
        kernel_size=kernel_size,
        strides=1,
        padding='same',  # Equivalent to padding=2 for kernel_size=5
        use_bias=True,
        name='spatial_conv'
    )(spatial_pool)

    sigmoid_1: tf.keras.layers.Layer = tf.keras.layers.Lambda(lambda y: tf.sigmoid(y))(conv2d_1)
    return tf.keras.layers.Multiply()([inputs, sigmoid_1])

def channel_attention_module(inputs: tf.keras.layers.Layer, filters:int=64, kernel_size:int=1, reduction: int = 8) -> tf.keras.layers.Layer:
    """
        Channel Attention Module (CAM)

        This module generates a channel-wise attention vector by applying
        global average pooling (GAP) followed by two fully-connected (1×1 Conv)
        layers with reduction to capture inter-channel dependencies.
        The final sigmoid-activated vector rescales each channel based on
        its learned importance.

        Args:
            inputs (tf.keras.layers.Layer):
                Input feature map of shape (H, W, C).
            filters (int):
                Unused (for interface consistency). Default = 64.
            kernel_size (int):
                Kernel size for both 1×1 convolutions. Default = 1.
            reduction (int):
                Channel reduction ratio for bottleneck (C → C/reduction). Default = 8.

        Returns:
            tf.keras.layers.Layer:
                Output tensor after applying channel attention, computed as:
                `inputs * channel_mask`
    """
    input_channels: int = inputs.shape[-1]
    gap: tf.keras.layers.Layer = tf.keras.layers.Lambda(lambda y: tf.reduce_mean(y, keepdims=True, axis=[1,2]))(inputs)
    conv2d_1: tf.keras.layers.Layer = tf.keras.layers.Conv2D(
        padding='same',
        filters=input_channels // reduction,
        use_bias=True,
        kernel_size=kernel_size)(gap)
    relu_1: tf.keras.layers.Layer = tf.keras.layers.ReLU()(conv2d_1)
    conv2d_2: tf.keras.layers.Layer = tf.keras.layers.Conv2D(padding='same', filters=input_channels, kernel_size=kernel_size)(relu_1)
    sigmoid_1: tf.keras.layers.Layer = tf.keras.layers.Lambda(lambda y: tf.sigmoid(y))(conv2d_2)
    return tf.keras.layers.Multiply()([inputs, sigmoid_1])


def spatial_and_channel_attention_module(inputs: tf.keras.layers.Layer, filters:int=64, kernel_size:int=3) -> tf.keras.layers.Layer:
    """
     Combined Spatial + Channel Attention Module (SCAM)

     This module fuses both spatial attention (SAM) and channel attention (CAM)
     to enhance the feature representation.
     It first processes the input with two convolutional layers, then
     independently applies SAM and CAM to capture both spatial and channel-wise
     importance. The outputs of SAM and CAM are concatenated and passed through
     a fusion convolution.
     Finally, a residual connection adds the original input to the output for
     stable training and better gradient flow.

     Args:
         inputs (tf.keras.layers.Layer):
             Input feature map of shape (H, W, C).
         filters (int):
             Number of filters for intermediate convolution layers. Default = 64.
         kernel_size (int):
             Kernel size for all internal convolutions. Default = 3.

     Returns:
         tf.keras.layers.Layer:
             Output tensor after spatial + channel attention with residual connection.
    """
    x: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', use_bias=True, kernel_size=kernel_size)(inputs)
    x: tf.keras.layers.Layer = tf.keras.layers.PReLU()(x)
    x: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', use_bias=True, kernel_size=kernel_size)(x)
    sam_out: tf.keras.layers.Layer = spatial_attention_module(x)
    cam_out: tf.keras.layers.Layer = channel_attention_module(x)
    scam: tf.keras.layers.Layer = tf.keras.layers.Concatenate(axis=-1)([sam_out, cam_out])
    x: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=True)(scam)

    return tf.keras.layers.Add()([inputs, x])

def upper_sub_network(inputs: tf.keras.layers.Layer, filters: int = 64, kernel_size: int = 3, out_channels: int = 3) -> tf.keras.layers.Layer:
    """
       Upper Sub-Network (U-Net–like Encoder–Decoder Module)

       This module acts as a shallow U-Net–style encoder–decoder block with
       skip connections, max-pooling for downsampling, and bilinear upsampling
       for reconstruction. It extracts hierarchical features, compresses them,
       and then reconstructs enhanced representations while retaining fine
       details through skip additions.

       Architecture Summary:
           • 4 convolutional layers → Downsample ×2 (via MaxPool)
           • 3 convolutional layers → Downsample ×2
           • 2 convolutional layers → UpSampling → skip-add with encoder
           • 2 convolutional layers → UpSampling → skip-add with encoder
           • Convolution → output feature map

       Args:
           inputs (tf.keras.layers.Layer):
               Input feature tensor of shape (H, W, C).
           filters (int):
               Number of convolution filters used throughout the subnetwork. Default = 64.
           kernel_size (int):
               Kernel size for all convolutional layers. Default = 3.
           out_channels (int):
               Number of channels in the final output layer. Default = 3.

       Returns:
           tf.keras.layers.Layer:
               Output tensor after encoder–decoder processing.
    """
    x1: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size, use_bias=True)(inputs)
    x1: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x1)
    x1: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x1)

    x2: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size, use_bias=True)(x1)
    x2: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x2)
    x2: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x2)

    x3: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size,use_bias=True)(x2)
    x3: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x3)
    x3: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x3)

    x4: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size,use_bias=True)(x3)
    x4: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x4)
    x4: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x4)

    mp_1: tf.keras.layers.Layer = tf.keras.layers.MaxPool2D()(x4)
    x5: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size, use_bias=True)(mp_1)
    x5: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x5)
    x5: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x5)

    x6: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size,use_bias=True)(x5)
    x6: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x6)
    x6: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x6)

    x7: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size, use_bias=True)(x6)
    x7: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x7)
    x7: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x7)

    mp_2: tf.keras.layers.Layer = tf.keras.layers.MaxPool2D()(x7)

    x8: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size,use_bias=True)(mp_2)
    x8: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x8)
    x8: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x8)

    x9: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size,use_bias=True)(x8)
    x9: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x9)
    x9: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x9)

    up_1: tf.keras.layers.Layer = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x9)

    x10: tf.keras.layers.Layer = tf.keras.layers.Add()([up_1, x7])
    x10: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x10)
    x10: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x10)

    x11: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=True, padding='same')(x10)
    x11: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x11)
    x11: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x11)

    x12: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=True,padding='same')(x11)
    x12: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x12)
    x12: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x12)

    up_2: tf.keras.layers.Layer = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x12)

    x13: tf.keras.layers.Layer = tf.keras.layers.Add()([up_2, x4])
    x13: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x13)
    x13: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x13)

    x14: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=True,padding='same')(x13)
    x14: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x14)
    x14: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x14)

    x15: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=True,padding='same')(x14)
    x15: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x15)
    x15: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x15)

    x16: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding='same', use_bias=True)(x15)

    return x16

def lower_sub_network(inputs: tf.keras.layers.Layer, filters: int = 64, kernel_size: int = 3, out_channels: int = 3) -> tf.keras.layers.Layer:
    """
       Lower Sub-Network (Multi-Dilation Residual Pyramid Module)

       This module extracts multi-scale contextual information using a
       progressive stack of dilated convolutions (dilation 1 → 8), followed
       by a mirrored contraction path (8 → 1) with residual additions.
       The design allows extremely large effective receptive fields without
       losing spatial resolution, making it ideal for capturing large-area
       structures or long-range image dependencies.

       Architecture Summary:
           • Dilation path: 1, 2, 3, 4, 5, 6, 7, 8
           • Reverse dilation path with residual connections:
                 8→7 skip → conv(d6)
                 6 skip → conv(d5)
                 …
                 2 skip → conv(d1)
           • Final convolution → output channels

       This structure acts like a residual dilated pyramid.

       Args:
           inputs (tf.keras.layers.Layer):
               Input feature map of shape (H, W, C).
           filters (int):
               Number of convolutional filters for all layers. Default = 64.
           kernel_size (int):
               Kernel size for all dilated convolutions. Default = 3.
           out_channels (int):
               Number of output channels in the final convolution. Default = 3.

       Returns:
           tf.keras.layers.Layer:
               Output tensor representing multi-scale aggregated features.
    """
    x1: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=1, use_bias=True)(inputs)
    x1: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x1)
    x1: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x1)

    x2: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=2, use_bias=True)(x1)
    x2: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x2)
    x2: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x2)

    x3: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=3, use_bias=True)(x2)
    x4: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x3)
    x3: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x3)

    x4: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=4, use_bias=True)(x3)
    x4: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x4)
    x4: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x4)

    x5: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=5, use_bias=True)(x4)
    x5: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x5)
    x5: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x5)

    x6: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=6, use_bias=True)(x5)
    x6: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x6)
    x6: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x6)

    x7: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=7, use_bias=True)(x6)
    x7: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x7)
    x7: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x7)

    x8: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=8, use_bias=True)(x7)
    x8: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x8)
    x8: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x8)

    x9: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=7, use_bias=True)(x8)
    x9: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x9)
    x9: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x9)

    x10: tf.keras.layers.Layer = tf.keras.layers.Add()([x9, x7])
    x10: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=6, use_bias=True)(x10)
    x10: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x10)
    x10: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x10)

    x11: tf.keras.layers.Layer = tf.keras.layers.Add()([x10, x6])
    x11: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=5, use_bias=True)(x11)
    x11: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x11)
    x11: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x11)

    x12: tf.keras.layers.Layer = tf.keras.layers.Add()([x11, x5])
    x12: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=4, use_bias=True)(x12)
    x12: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x12)
    x12: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x12)

    x13: tf.keras.layers.Layer = tf.keras.layers.Add()([x12, x4])
    x13: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=3, use_bias=True)(x13)
    x13: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x13)
    x13: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x13)

    x14: tf.keras.layers.Layer = tf.keras.layers.Add()([x13, x3])
    x14: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=2, use_bias=True)(x14)
    x14: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x14)
    x14: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x14)

    x15: tf.keras.layers.Layer = tf.keras.layers.Add()([x14, x2])
    x15: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=1, use_bias=True)(x15)
    x15: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x15)
    x15: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x15)

    x16: tf.keras.layers.Layer = tf.keras.layers.Add()([x15, x1])
    x16: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding='same',
                                                       dilation_rate=1, use_bias=True)(x16)
    # x16: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()(x16)
    # x16: tf.keras.layers.Layer = tf.keras.layers.ReLU()(x16)

    return x16




def dca_net(input_shape:tuple=(128, 128, 3), filters: int=64, kernel_size: int=3, out_channels: int=3) -> tf.keras.Model:
    """
        DCA-Net: Dual Convolutional Neural Network with Attention for Image Blind Denoising

        This architecture integrates:
            ✔ Noise Estimation Network
            ✔ Dual-branch restoration subnetworks
            ✔ Spatial–Channel Attention Module (SCAM)
            ✔ Residual learning at multiple levels

        Workflow:
            1. Noise Estimation:
                A dedicated CNN predicts a pixel-wise noise map.
            2. Concatenation:
                Input image + predicted noise → combined noisy representation.
            3. Attention:
                Spatial+Channel attention enhances significant features.
            4. Dual Branches:
                • Upper Sub-Network   → U-Net style local detail restoration
                • Lower Sub-Network   → Dilated pyramid for global context
            5. Output Fusion:
                X = upper_output + input
                Y = lower_output + input
                Z = concat(X, Y)
            6. Final reconstruction:
                Conv → output_image
                Residual addition produces the final restored image.

        Model Outputs:
            • output_image  — final denoised image
            • conv_tail     — fused output before final residual
            • noise_map     — estimated noise distribution

        Args:
            input_shape (tuple):
                Shape of input images (default: 128×128×3).
            filters (int):
                Base number of convolutional filters. Default = 64.
            kernel_size (int):
                Kernel size for shared convolutions. Default = 3.
            out_channels (int):
                Output image channels (typically RGB=3). Default = 3.

        Returns:
            tf.keras.Model:
                A multi-output Keras model implementing DCA-Net.
    """
    inputs: tf.keras.layers.Layer = tf.keras.layers.Input(shape=input_shape)
    noise_estimation_output: tf.keras.layers.Layer = noise_estimation_network(input_layer=inputs)

    input_noise_concatenated: tf.keras.layers.Layer = tf.keras.layers.Concatenate(axis=-1)([inputs, noise_estimation_output])
    input_noise: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel_size)(input_noise_concatenated)
    scam_output: tf.keras.layers.Layer = spatial_and_channel_attention_module(inputs=input_noise)

    upper_sub_network_output: tf.keras.layers.Layer = upper_sub_network(inputs=scam_output)
    X: tf.keras.layers.Layer = tf.keras.layers.Add()([upper_sub_network_output, inputs])

    lower_sub_network_output: tf.keras.layers.Layer = lower_sub_network(inputs=scam_output)
    Y: tf.keras.layers.Layer = tf.keras.layers.Add()([lower_sub_network_output, inputs])

    Z: tf.keras.layers.Layer = tf.keras.layers.Concatenate(axis=-1)([X, Y])

    conv_tail: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding='same', use_bias=True)(Z)
    output: tf.keras.layers.Layer = tf.keras.layers.Add()([conv_tail, inputs])
    model = tf.keras.Model(inputs=inputs, outputs=[output, conv_tail, noise_estimation_output])

    # print(model.summary())
    return model

if __name__ == '__main__':
    dca_net((256, 256, 3))