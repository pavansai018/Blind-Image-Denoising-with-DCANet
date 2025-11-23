import tensorflow as tf

def noise_estimation_network(input_layer:tf.keras.layers.Layer=tf.keras.layers.Input(shape=(128, 128, 3)),
                             filters:int = 64, kernel_size:int = 3, out_channels: int = 3):

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',)(input_layer)
    x = tf.keras.layers.ReLU()(x)
    for i in range(5):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding='same', )(x)
    outputs = tf.keras.layers.Activation('tanh')(x)

    return outputs

def spatial_attention_module(inputs: tf.keras.layers.Layer, filters:int=64, kernel_size:int=5):
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

def channel_attention_module(inputs: tf.keras.layers.Layer, filters:int=64, kernel_size:int=1, reduction: int = 8):
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


def spatial_and_channel_attention_module(inputs: tf.keras.layers.Layer, filters:int=64, kernel_size:int=3):
    x: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', use_bias=True, kernel_size=kernel_size)(inputs)
    x: tf.keras.layers.Layer = tf.keras.layers.PReLU()(x)
    x: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, padding='same', use_bias=True, kernel_size=kernel_size)(x)
    sam_out: tf.keras.layers.Layer = spatial_attention_module(x)
    cam_out: tf.keras.layers.Layer = channel_attention_module(x)
    scam: tf.keras.layers.Layer = tf.keras.layers.Concatenate(axis=-1)([sam_out, cam_out])
    x: tf.keras.layers.Layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=True)(scam)

    return tf.keras.layers.Add()([inputs, x])

def upper_sub_network(inputs: tf.keras.layers.Layer, filters: int = 64, kernel_size: int = 3, out_channels: int = 3):

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

def lower_sub_network(inputs: tf.keras.layers.Layer, filters: int = 64, kernel_size: int = 3, out_channels: int = 3):
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




def dca_net(input_shape:tuple=(128, 128, 3), filters: int=64, kernel_size: int=3, out_channels: int=3):
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

    print(model.summary())
    return model

if __name__ == '__main__':
    dca_net((256, 256, 3))