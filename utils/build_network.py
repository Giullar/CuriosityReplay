import tensorflow as tf
from tensorflow import keras


def build_network(input_shape, n_outputs, conv=[], hidden_dense=[], final_layer=None):
    w_initializer = "glorot_uniform"
    b_initializer = "zeros"
    inputs = x = keras.Input(shape=input_shape)
    
    for (filters, kernel_size, stride) in conv:
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, activation='elu', padding='same',
                                data_format="channels_first",
                                kernel_initializer=w_initializer,
                                bias_initializer=b_initializer)(x)
        
    if len(conv) > 0:
        x = keras.layers.Flatten()(x)
    
    for units in hidden_dense:
        x = keras.layers.Dense(units, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
    
    if final_layer == "dueling":
        state_value = keras.layers.Dense(1, activation=None, kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        advantages = keras.layers.Dense(n_outputs, activation=None, kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        advantages_mean = tf.reduce_mean(advantages, 1)
        advantages_centered = advantages - tf.expand_dims(advantages_mean, 1)
        outputs = state_value + advantages_centered
    elif final_layer == "dense":
        outputs = keras.layers.Dense(n_outputs, activation=None, kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
    elif final_layer == "dense_softmax":
        outputs = keras.layers.Dense(n_outputs, activation="softmax", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        
    elif final_layer is None:
        outputs = x
    else:
        raise Exception("Incorrect final layer specification")
    
    model = tf.keras.Model(inputs=inputs, outputs=[outputs])
    #display(keras.utils.plot_model(model, show_shapes=True))
    return model
