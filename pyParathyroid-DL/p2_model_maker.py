# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:52:08 2021

@author: John
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
#in_shape = (300, 300, 3)
#classes = 2

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))




def build_siamese_model_old(inputShape, embeddingDim=200):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)
	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = BatchNormalization()(x)
	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = BatchNormalization()(x)

	x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)
	# build the model
	model = tf.keras.Model(inputs, outputs)
	# return the model to the calling function
	return model


def build_siamese_vgg(inputShape, embeddingDim=750):

       base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputShape, pooling=None, classes=1)

       layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

       for layer in base_model.layers:
           layer.trainable = False
       for layer in base_model.layers[19:]:
           layer.trainable = True
       #x = layer_dict['block2_pool'].output
       x = layer_dict['block5_pool'].output
       #perhaps train the first layers and completelely disconnect the last
       
       # x = tf.keras.layers.Dense(1000, activation='relu')(x)
       # #x = tf.keras.layers.BatchNormalization()(x)
       #x = tf.keras.layers.Dropout(0.5)(x)       
       # x = tf.keras.layers.Dense(500, activation='relu')(x)
       # #x = tf.keras.layers.BatchNormalization()(x)
       # x = tf.keras.layers.Dropout(0.5)(x)        
       x = GlobalAveragePooling2D()(x)
       outputs = Dense(embeddingDim)(x)

       model = tf.keras.Model(base_model.input, outputs)

       return model





def build_siamese_xception(inputShape, embeddingDim=500):

    base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    classes=1)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    for layer in base_model.layers:
        layer.trainable = True
    

    for layer in base_model.layers[:-5]:
        layer.trainable = False
        
    
    x = layer_dict['block14_sepconv2_act'].output
    
    
    x= tf.keras.layers.GlobalAveragePooling2D()(x)     
    outputs = Dense(embeddingDim)(x)
    
    model = tf.keras.Model(base_model.input, outputs)

    return model


def build_siamese_efficient(inputShape, embeddingDim=500):

    base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    classes=1)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    for layer in base_model.layers:
        layer.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    x = layer_dict['top_activation'].output 
    
    
    x= tf.keras.layers.GlobalAveragePooling2D()(x)     
    outputs = Dense(embeddingDim)(x)
    
    model = tf.keras.Model(base_model.input, outputs)

    return model



def build_siamese_inception(inputShape, embeddingDim=500):

    base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    classes=1)
    
    #for layer in base_model.layers:
        #print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    for layer in base_model.layers:
        layer.trainable = True

    for layer in base_model.layers[:-25]:
        layer.trainable = False

    x = layer_dict['mixed10'].output 
    
    
    x= tf.keras.layers.GlobalAveragePooling2D()(x)     
    outputs = Dense(embeddingDim)(x)
    
    model = tf.keras.Model(base_model.input, outputs)

    return model


def siamese_network(in_shape, tune, classes):
    
    one = in_shape[0]
    zero = in_shape[1]
    in_shape = (zero,one,in_shape[2])
    
    imgA = Input(shape=in_shape)
    imgB = Input(shape=in_shape)
    
    siamer = build_siamese_vgg(inputShape=in_shape)
    siam_1 = siamer(imgA)
    siam_2 = siamer(imgB)
    
    distance = Lambda(euclidean_distance)([siam_1, siam_2])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = tf.keras.Model(inputs=[imgA, imgB], outputs=outputs)
    
    #opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False, name="SGD")
    #opt = tf.keras.optimizers.Adam(
    #learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
    model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["binary_accuracy"])
    return model


def siamese_triplet(in_shape, tune, classes):
    
    one = in_shape[0]
    zero = in_shape[1]
    in_shape = (zero,one,in_shape[2])
    
    imgA = Input(shape=in_shape)
    imgB = Input(shape=in_shape)
    imgC = Input(shape=in_shape)
    
    siamer = build_siamese_vgg(inputShape=in_shape)
    siam_1 = siamer(imgA)
    siam_2 = siamer(imgB)
    siam_3 = siamer(imgC)
    
    distance_el = Lambda(euclidean_distance)([siam_1, siam_2])
    distance_es = Lambda(euclidean_distance)([siam_1, siam_3])
    distances = tf.keras.layers.concatenate([distance_el, distance_es])
    outputs = Dense(1, activation="sigmoid")(distances)
    model = tf.keras.Model(inputs=[imgA, imgB,imgC], outputs=outputs)
    
    #opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False, name="SGD")
    #opt = tf.keras.optimizers.Adam(
    #learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
    model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["binary_accuracy"])
    return model


def make_lb(inputs):
    
    
    
    mcb = tf.keras.layers.Conv2D(3, 1, strides=1, padding="valid")(inputs)
    mcb = tf.keras.layers.BatchNormalization()(mcb)
    #mcb = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(mcb)
    
    mcb_a = tf.keras.layers.Conv2D(3, 3, strides=1, padding="same")(mcb)
    mcb_a = tf.keras.layers.BatchNormalization()(mcb_a)
    
    mcb_b = tf.keras.layers.Conv2D(3, 5, strides=1, padding="same")(mcb)
    mcb_b = tf.keras.layers.BatchNormalization()(mcb_b)
    
    mcb_c = tf.keras.layers.Conv2D(3, 7, strides=1, padding="same")(mcb)
    mcb_c = tf.keras.layers.BatchNormalization()(mcb_c)
    
    
    concat = tf.keras.layers.concatenate([mcb_a, mcb_b, mcb_c], axis=-1)
    mcb_final_conv = tf.keras.layers.Conv2D(3, 1, strides=1, padding="same")(concat)
    mcb_final_conv = tf.keras.layers.BatchNormalization()(mcb_final_conv)
    
    mb = tf.keras.layers.Conv2D(3, 1, strides=1, padding="same")(inputs)
    mb = tf.keras.layers.BatchNormalization()(mb)
    #mb = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(mb)
    
    concat_final = tf.keras.layers.concatenate([mcb_final_conv, mb], axis=-1)
    final = tf.keras.layers.Conv2D(3, 1, strides=1, padding="valid")(concat_final)
    
    return final
    


def make_lb_cnn(in_shape, tune, classes):
    
    inputs = tf.keras.Input(shape=in_shape)
    
    
    lb = make_lb(inputs)
    pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lb)
    lb = make_lb(pool)
    pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lb)
    lb = make_lb(pool)
    pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lb)
    lb = make_lb(pool)
    pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lb)
    lb = make_lb(pool)
    pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lb)
    lb = make_lb(pool)
    pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lb)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(lb)
   # x =  tf.keras.layers.Flatten()(pool)
    #x = tf.keras.layers.Dropout(0.5)(x)
    #x = tf.keras.layers.Dense(2, activation="relu")(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dense(2, activation="relu")(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    
    
    #x = layers.SeparableConv2D(size, 3, padding="same")(x)
    
    #model.summary()  
    return model




def make_lb_multi(in_shape, tune, classes):
    
    inputs_a = tf.keras.Input(shape=in_shape)
    inputs_b = tf.keras.Input(shape=in_shape)
    inputs_c = tf.keras.Input(shape=in_shape)
    
    lbA = make_lb(inputs_a)
    poolA = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbA)
    lbA = make_lb(poolA)
    poolA = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbA)
    lbA = make_lb(poolA)
    poolA = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbA)
    lbA = make_lb(poolA)
    
    lbB = make_lb(inputs_b)
    poolB = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbB)
    lbB = make_lb(poolB)
    poolB = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbB)
    lbB = make_lb(poolB)
    poolB = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbB)   
    lbB = make_lb(poolB)
    
    
    lbC = make_lb(inputs_c)
    poolC = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbC)
    lbC = make_lb(poolC)
    poolC = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbC)
    lbC = make_lb(poolC)
    poolC = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbC)
    lbC = make_lb(poolC)
    
    first_level = tf.keras.layers.concatenate([lbA, lbB, lbC], axis=1)
    #pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(first_level)
    #lb2 = make_lb(pool)
    pool2= tf.keras.layers.GlobalAveragePooling2D()(first_level)
    
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(pool2)
    
    model = tf.keras.Model(inputs=[inputs_a,inputs_b,inputs_c], outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    

def make_lb_multi_2_img(in_shape, tune, classes):
    
    inputs_a = tf.keras.Input(shape=in_shape)
    inputs_b = tf.keras.Input(shape=in_shape)
    
    lbA = make_lb(inputs_a)
    poolA = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbA)
    lbA = make_lb(poolA)
    poolA = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbA)
    lbA = make_lb(poolA)
    poolA = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbA)
    lbA = make_lb(poolA)
    
    lbB = make_lb(inputs_b)
    poolB = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbB)
    lbB = make_lb(poolB)
    poolB = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbB)
    lbB = make_lb(poolB)
    poolB = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(lbB)   
    lbB = make_lb(poolB)
    
    
    first_level = tf.keras.layers.concatenate([lbA, lbB], axis=1)
    #pool = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(first_level)
    #lb2 = make_lb(pool)
    pool2= tf.keras.layers.GlobalAveragePooling2D()(first_level)
    
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(pool2)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = tf.keras.Model(inputs=[inputs_a,inputs_b], outputs = outputs)
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def make_vgg (in_shape, tune, classes): #tune = 0 is off the self, tune = 1 is scratch, tune 
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune == 20:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[19:]:
            layer.trainable = True
    #base_model.summary()
  
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_lvgg (in_shape, tune, classes):
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = True
    #base_model.summary()
    
    # early2 = layer_dict['block2_pool'].output 
    # #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    # early2 = tf.keras.layers.BatchNormalization()(early2)
    # early2 = tf.keras.layers.Dropout(0.5)(early2)
    # early2= tf.keras.layers.GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_conv4'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    #early3 = tf.keras.layers.Dropout(0.2)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_conv4'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    #early4 = tf.keras.layers.Dropout(0.2)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(550, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.summary()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_xception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['block14_sepconv2'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 

def make_inception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    #for layer in base_model.layers:
        #print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['mixed10'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    #print("[INFO] Model Compiled!")
    return model 

def make_resnet (in_shape, tune, classes):
    
    base_model = tf.keras.applications.ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['post_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 


def make_mobile (in_shape, tune, classes):
    
    base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['out_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 


def make_dense (in_shape, tune, classes):
    
    base_model = tf.keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 

def make_eff (in_shape, tune, classes):
    
    base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_self (in_shape, tune, classes):
    
    inputs = tf.keras.Input(shape=in_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    
    
    #x = layers.SeparableConv2D(size, 3, padding="same")(x)
    
    model.summary()
    return model

def make_multi_self (in_shape, tune, classes):
    
    inputs_a = tf.keras.Input(shape=in_shape)
    x = tf.keras.layers.BatchNormalization()(inputs_a)
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    
    inputs_b = tf.keras.Input(shape=in_shape)
    y = tf.keras.layers.BatchNormalization()(inputs_b)
    y = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(y)
    y = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(y)
    y = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(y)
    y = tf.keras.layers.GlobalMaxPooling2D()(y)

    inputs_c = tf.keras.Input(shape=in_shape)
    z = tf.keras.layers.BatchNormalization()(inputs_c)
    z = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    z = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(z)
    z = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    z = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(z)
    z = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    z = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(z)
    z = tf.keras.layers.GlobalMaxPooling2D()(z)    
    
    
    output = tf.keras.layers.concatenate([x, y, z], axis=1)
    
    
    
    
    exodus = tf.keras.layers.Dropout(0.5)(output)
    exodus = tf.keras.layers.Dense(1200, activation="relu")(exodus)
    exodus = tf.keras.layers.BatchNormalization()(exodus)
    exodus = tf.keras.layers.Dense(512, activation="relu")(exodus)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(exodus)
    
    model = tf.keras.Model(inputs=[inputs_a,inputs_b,inputs_c], outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    
    
    #x = layers.SeparableConv2D(size, 3, padding="same")(x)
    
    model.summary()
    return model


def make_3_vggs(in_shape, tune, classes):
    
    base_model_early = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_early.layers:
        layer._name = layer._name + str('_E')  
    
    layer_dict = dict([(layer.name, layer) for layer in base_model_early.layers])

    for layer in base_model_early.layers:
        layer.trainable = False
    for layer in base_model_early.layers[28:]:
        layer.trainable = True
        
    early3 = layer_dict['block3_pool_E'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    #early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool_E'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    #early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3_E'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    

    

    base_model_late = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_late.layers:
        layer._name = layer._name + str('_L')  
    layer_dict_late = dict([(layer.name, layer) for layer in base_model_late.layers])
    for layer in base_model_late.layers:
        layer.trainable = False
    for layer in base_model_late.layers[28:]:
        layer.trainable = True
        
    early3_late = layer_dict_late['block3_pool_L'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3_late = tf.keras.layers.BatchNormalization()(early3_late)
    #early3_late = tf.keras.layers.Dropout(0.5)(early3_late)
    early3_late= tf.keras.layers.GlobalAveragePooling2D()(early3_late)    
        
    early4_late = layer_dict_late['block4_pool_L'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4_late = tf.keras.layers.BatchNormalization()(early4_late)
    #early4_late = tf.keras.layers.Dropout(0.5)(early4_late)
    early4_late = tf.keras.layers.GlobalAveragePooling2D()(early4_late)     
    
    y1 = layer_dict_late['block5_conv3_L'].output 
    y1= tf.keras.layers.GlobalAveragePooling2D()(y1)
    y = tf.keras.layers.concatenate([y1, early4_late, early3_late], axis=-1)  
    
 

    base_model_sub = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    
    for layer in base_model_sub.layers:
        layer._name = layer._name + str('_S')  
    
    layer_dict_sub = dict([(layer.name, layer) for layer in base_model_sub.layers])
    for layer in base_model_sub.layers:
        layer.trainable = False
    for layer in base_model_sub.layers[28:]:
        layer.trainable = True
        
    early3_sub = layer_dict_sub['block3_pool_S'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3_sub = tf.keras.layers.BatchNormalization()(early3_sub)
    #early3_sub = tf.keras.layers.Dropout(0.5)(early3_sub)
    early3_sub = tf.keras.layers.GlobalAveragePooling2D()(early3_sub)    
        
    early4_sub = layer_dict_sub['block4_pool_S'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4_sub = tf.keras.layers.BatchNormalization()(early4_sub)
    #early4_sub = tf.keras.layers.Dropout(0.5)(early4_sub)
    early4_sub = tf.keras.layers.GlobalAveragePooling2D()(early4_sub)     
    
    z1 = layer_dict_sub['block5_conv3_S'].output 
    z1= tf.keras.layers.GlobalAveragePooling2D()(z1)
    z = tf.keras.layers.concatenate([z1, early4_sub, early3_sub], axis=-1) 

    
    exodus = tf.keras.layers.concatenate([x, y, z], axis=-1) 
    
    #exodus = tf.keras.layers.Dropout(0.5)(exodus)
    exodus = tf.keras.layers.Dense(750, activation="relu")(exodus)
    exodus = tf.keras.layers.Dropout(0.5)(exodus)
    exodus = tf.keras.layers.Dense(256, activation="relu")(exodus)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(exodus)    
    
    model = tf.keras.Model(inputs=[base_model_early.input,base_model_late.input,base_model_sub.input], outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    return model


def make_2_vggs(in_shape, tune, classes):
    
    base_model_early = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_early.layers:
        layer._name = layer._name + str('_E')  
    
    layer_dict = dict([(layer.name, layer) for layer in base_model_early.layers])

    for layer in base_model_early.layers:
        layer.trainable = False
    for layer in base_model_early.layers[19:]:
        layer.trainable = True
        
    early3 = layer_dict['block3_pool_E'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool_E'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3_E'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    

    

    base_model_late = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_late.layers:
        layer._name = layer._name + str('_L')  
    layer_dict_late = dict([(layer.name, layer) for layer in base_model_late.layers])
    for layer in base_model_late.layers:
        layer.trainable = False
    for layer in base_model_late.layers[19:]:
        layer.trainable = True
        
    early3_late = layer_dict_late['block3_pool_L'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3_late = tf.keras.layers.BatchNormalization()(early3_late)
    early3_late = tf.keras.layers.Dropout(0.5)(early3_late)
    early3_late= tf.keras.layers.GlobalAveragePooling2D()(early3_late)    
        
    early4_late = layer_dict_late['block4_pool_L'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4_late = tf.keras.layers.BatchNormalization()(early4_late)
    early4_late = tf.keras.layers.Dropout(0.5)(early4_late)
    early4_late = tf.keras.layers.GlobalAveragePooling2D()(early4_late)     
    
    y1 = layer_dict_late['block5_conv3_L'].output 
    y1= tf.keras.layers.GlobalAveragePooling2D()(y1)
    y = tf.keras.layers.concatenate([y1, early4_late, early3_late], axis=-1)  
    
 

    
    exodus = tf.keras.layers.concatenate([x, y], axis=-1)
    #exodus= tf.keras.layers.GlobalAveragePooling2D()(exodus)

    outputs = tf.keras.layers.Dense(classes, activation="softmax")(exodus)    
    
    model = tf.keras.Model(inputs=[base_model_early.input,base_model_late.input], outputs = outputs)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    return model

def make_3_vggs_lstm(in_shape, tune, classes):
    
    base_model_early = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_early.layers:
        layer._name = layer._name + str('_E')  
    
    layer_dict = dict([(layer.name, layer) for layer in base_model_early.layers])

    for layer in base_model_early.layers:
        layer.trainable = False
    for layer in base_model_early.layers[19:]:
        layer.trainable = False
        
    early3 = layer_dict['block3_pool_E'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool_E'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3_E'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    x = layer_dict['block5_conv3_E'].output
    x= tf.keras.layers.GlobalAveragePooling2D()(x)
    

    base_model_late = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_late.layers:
        layer._name = layer._name + str('_L')  
    layer_dict_late = dict([(layer.name, layer) for layer in base_model_late.layers])
    for layer in base_model_late.layers:
        layer.trainable = False
    for layer in base_model_late.layers[10:20]:
        layer.trainable = False
        
    early3_late = layer_dict_late['block3_pool_L'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3_late = tf.keras.layers.BatchNormalization()(early3_late)
    early3_late = tf.keras.layers.Dropout(0.5)(early3_late)
    early3_late= tf.keras.layers.GlobalAveragePooling2D()(early3_late)    
        
    early4_late = layer_dict_late['block4_pool_L'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4_late = tf.keras.layers.BatchNormalization()(early4_late)
    early4_late = tf.keras.layers.Dropout(0.5)(early4_late)
    early4_late = tf.keras.layers.GlobalAveragePooling2D()(early4_late)     
    
    y1 = layer_dict_late['block5_conv3_L'].output 
    y1= tf.keras.layers.GlobalAveragePooling2D()(y1)
    #y = tf.keras.layers.concatenate([y1, early4_late, early3_late], axis=-1)  
    y = layer_dict_late['block5_conv3_L'].output
    y= tf.keras.layers.GlobalAveragePooling2D()(y)

    base_model_sub = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    
    for layer in base_model_sub.layers:
        layer._name = layer._name + str('_S')  
    
    layer_dict_sub = dict([(layer.name, layer) for layer in base_model_sub.layers])
    for layer in base_model_sub.layers:
        layer.trainable = False
    for layer in base_model_sub.layers[19:]:
        layer.trainable = False
        
    early3_sub = layer_dict_sub['block3_pool_S'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3_sub = tf.keras.layers.BatchNormalization()(early3_sub)
    early3_sub = tf.keras.layers.Dropout(0.5)(early3_sub)
    early3_sub = tf.keras.layers.GlobalAveragePooling2D()(early3_sub)    
        
    early4_sub = layer_dict_sub['block4_pool_S'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4_sub = tf.keras.layers.BatchNormalization()(early4_sub)
    early4_sub = tf.keras.layers.Dropout(0.5)(early4_sub)
    early4_sub = tf.keras.layers.GlobalAveragePooling2D()(early4_sub)     
    
    z1 = layer_dict_sub['block5_conv3_S'].output 
    z1= tf.keras.layers.GlobalAveragePooling2D()(z1)
    z = tf.keras.layers.concatenate([z1, early4_sub, early3_sub], axis=-1) 
    z = layer_dict_sub['block5_conv3_S'].output
    z= tf.keras.layers.GlobalAveragePooling2D()(z)
    
    
    exodus = tf.keras.layers.concatenate([x, y, z], axis=-1)
    exodus = tf.keras.layers.Dense(100, activation='relu')(exodus)
    exodus = tf.keras.layers.Reshape((-1, 100))(exodus)
    

    
    exodus = tf.keras.layers.LSTM(100, return_sequences=True)(exodus)
    exodus = tf.keras.layers.Dropout(0.5)(exodus)
    exodus = tf.keras.layers.LSTM(25, return_sequences=False)(exodus)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(exodus)  
    
    model = tf.keras.Model(inputs=[base_model_early.input,base_model_late.input,base_model_sub.input], outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    return model






def make_3d_bench(in_shape, tune, classes):
    
    model_input_shape = (3, in_shape[0],in_shape[1], in_shape[2] )
    
    def conv3D (x,filters,bn,maxp=0,rdmax=0,drop=True,DepthPool=False):
        
        x  = tf.keras.layers.Conv3D(filters, (1, 3,3), padding='valid', activation='relu')(x)
        
        
        if maxp ==1 and DepthPool==False:
            x = tf.keras.layers.MaxPooling3D((1,2, 2),padding='valid')(x)
        if  DepthPool==True:   
            x = tf.keras.layers.MaxPooling3D((2,2, 2),padding='valid')(x)
            
        if drop==True:
            x = tf.keras.layers.Dropout(0.5)(x)
            
        if bn==1:
            x = tf.keras.layers.BatchNormalization(axis=-1)(x)
            
        if rdmax == 1:
            x = tf.keras.layers.MaxPooling3D((2,2, 2),padding='valid')(x)
        return x
    
    input_img =  tf.keras.layers.Input(shape=model_input_shape) 
    
    x = conv3D (input_img,filters=16,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    x = conv3D (x,16,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    
    y = conv3D (x,32,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    y = conv3D (y,32,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    #y = conv3D (y,64,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    
    z = conv3D (y,64,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    z = conv3D (z,64,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)

    # w = conv3D (z,512,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    # w = conv3D (w,512,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    # w = conv3D (w,512,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    # w = conv3D (w,512,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    #k = conv3D (w,128,0,0)
    #m = conv3D (l,1024,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    
    #o = conv3D (m,512,1,1)
    #i = conv3D (o,512,1,0)
    
    #a = GlobalAveragePooling3D()(y)
    #b = GlobalAveragePooling3D()(z)
    c =  tf.keras.layers.GlobalAveragePooling3D()(z)
    #d = GlobalAveragePooling3D()(l)
    #e = GlobalAveragePooling3D()(i)
    #f = GlobalAveragePooling3D()(m)
    
    #n = keras.layers.concatenate([c,b,d], axis=-1)
    
    
    
    n = tf.keras.layers.Dense(500, activation='relu')(c)
    n = tf.keras.layers.Dropout(0.5)(n)

    n = tf.keras.layers.Dense(2, activation='softmax')(n)
    model = tf.keras.Model(inputs=input_img, outputs=n)
    
    #opt = SGD(lr=0.01)
    
    #opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    
#import pydot

    return model




def cnn_lstm(in_shape, tune, classes):
    
    '''CNN'''
    def cnn(in_shape, tune, classes):
        base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
        #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        #base_model.summary()
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[18:]:
            layer.trainable = False
      
        early3 = layer_dict['block3_pool'].output   
        early3 = tf.keras.layers.BatchNormalization()(early3)
        early3 = tf.keras.layers.Dropout(0.5)(early3)
        early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
            
        early4 = layer_dict['block4_pool'].output   
        early4 = tf.keras.layers.BatchNormalization()(early4)
        early4 = tf.keras.layers.Dropout(0.5)(early4)
        early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
        
        x1 = layer_dict['block5_conv3'].output 
        x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
        #x1 = Flatten()(x1)
        x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)
        exodus = tf.keras.layers.Dense(500, activation="relu")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=exodus)
        
        return model
    
    
    
    
    cnn_net = cnn(in_shape, tune, classes)
    
    input_layer = tf.keras.layers.Input(shape=(3, in_shape[0], in_shape[1], in_shape[2]))
    
    lstm_ip_layer = tf.keras.layers.TimeDistributed(cnn_net)(input_layer)
    
    lstm = tf.keras.layers.LSTM(500)(lstm_ip_layer)
    #lstm = tf.keras.layers.LSTM(200)(lstm)
    #lstm = tf.keras.layers.LSTM(50)(lstm)
    output = tf.keras.layers.Dense(units=2,activation='softmax')(lstm)
    model = tf.keras.Model([input_layer],output)

    optimizer = tf.keras.optimizers.Nadam(lr=0.002,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      schedule_decay=0.004)

    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model