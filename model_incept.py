from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K
from keras.optimizers import SGD
from data import create_data, data_generator

def run():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    top = base_model.output
    top = Flatten(input_shape=top.output_shape)(top)
    top = Dense(256, activation='tanh')(top)
    top = Dropout(0.5)(top)
    top = Dense(64, activation='tanh')(top)
    top = Dropout(0.5)(top)
    predictions = Dense(1)(top)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='mse')

    # train the model on the new data for a few epochs

    train_set, validation_set = create_data()
    model.fit_generator(
        data_generator(train_set),
        samples_per_epoch=len(train_set),
        nb_epoch=10,
        validation_data=data_generator(validation_set),
        nb_val_samples=len(validation_set))

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mse')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        data_generator(train_set),
        samples_per_epoch=len(train_set),
        nb_epoch=10,
        validation_data=data_generator(validation_set),
        nb_val_samples=len(validation_set))

if __name__ == "__main__":
    run()
