import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory


BATCH_SIZE = 32
IMG_SIZE = (160, 160)
AVAILABLE_MODELS = ["mobilenet", "resnet50", "resnet101", "resnet152", "inception_resnet_v2"]


def data_preparation():
    #data preparation
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')

    train_dataset = image_dataset_from_directory(train_dir,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE)

    validation_dataset = image_dataset_from_directory(validation_dir,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def main(model_name, dataset):
    #unpack dataset
    train_dataset, validation_dataset, test_dataset = dataset

    print("\n------ {} ------".format(model_name))
    if model_name == "mobilenet":
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif model_name.startswith("resnet"):
        preprocess_input = tf.keras.applications.resnet.preprocess_input
    elif model_name.startswith("inception"):
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
    else:
        raise NotImplementedError

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

    """Figure model here： ResNet50/101/152, MobileNet, InceptionResNetV2, ..."""

    # Create the base model from the pre-trained model
    IMG_SHAPE = IMG_SIZE + (3,)

    if model_name == "mobilenet":
        base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == "resnet50":
        base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == "resnet101":
        base_model = tf.keras.applications.ResNet101(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == "resnet152":
        base_model = tf.keras.applications.ResNet152(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == "inception_resnet_v2":
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    else:
        raise NotImplementedError

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    #print("feature batch shape: ", feature_batch.shape)

    base_model.trainable = True

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    #print("feature batch average shape: ",feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    #print("prediction batch shape: ", prediction_batch.shape)

    # build model
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=True)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # model.summary()

    print("trainable variables: ", len(model.trainable_variables))
    print('- Val')
    loss0, accuracy0 = model.evaluate(validation_dataset)
    print('- Train')
    history = model.fit(train_dataset,
                        epochs = 2,
                        validation_data=validation_dataset)



if __name__ == "__main__":
    all_dataset = data_preparation()

    for model in AVAILABLE_MODELS:
        main(model, all_dataset)
