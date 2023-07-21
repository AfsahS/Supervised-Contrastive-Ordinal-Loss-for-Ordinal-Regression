import tensorflow as tf
from efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.layers import *

def Attn_block(input_tensor):
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(input_tensor)test_stage1.py
    x_2 = tf.keras.layers.BatchNormalization()(x)
    x_2 = tf.keras.layers.Activation('relu')(x_2)
    x_2d = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x_2)
    x_2d = tf.keras.layers.BatchNormalization()(x_2d)
    x_2d = tf.keras.layers.Activation('relu')(x_2d)
    x_2d = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=None, padding='valid', )(x_2d)
    x_2sigmoid = tf.keras.layers.Activation('sigmoid')(x_2d)
    enc = tf.keras.layers.Multiply()([x, x_2sigmoid])
    return enc

def local_enc_proj():
    base_model = EfficientNetV2S(input_shape=(320, 320, 3), weights="imagenet", include_top=False)
    x = base_model.get_layer('top_conv').output
    enc = Attn_block(x)
    encout = GlobalAveragePooling2D()(enc)
    proj = Dense(1280, activation="relu")(encout)
    proj = Dropout(0.4)(proj)
    projout = Dense(128, activation="relu")(proj)
    model = tf.keras.Model(inputs=base_model.input, outputs=projout)
    return model

def global_enc_proj():
    base_model = EfficientNetV2S(input_shape=(320, 320, 3), weights="imagenet", include_top=False)
    x = base_model.get_layer('top_conv').output
    encout = GlobalAveragePooling2D()(x)
    proj = Dense(1280, activation="relu")(encout)
    proj = Dropout(0.4)(proj)
    projout = Dense(128, activation="relu")(proj)
    model = tf.keras.Model(inputs=base_model.input, outputs=projout)
    return model

