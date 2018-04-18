import os
import sys
import shutil
from tqdm import tqdm
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\ImageClassifier')

# If you wanna use pre trained model head on to
# https://github.com/llSourcell/how_to_make_an_image_classifier/blob/master/demo.ipynb

img_size = 150

# The path needs to be to a folder, that contains folders, that contains the images.
train_data_dir = './data/train/'
validation_data_dir = './data/validation/'

save_weight = False

epoch = 10
train_samples = 2048
validation_samples = 832


def create_small_conv_net():
    # Model architecture definition
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_size, img_size,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # Configure the model for training.
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def create_vgg_16_conv_net():
    # Model architecture definition
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_size, img_size,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # last layer same as our small conv_net
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # Configure the model for training.
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def split_data_into_cat_dog():

    train_data_list = os.listdir(train_data_dir)
    move_cat = False
    move_dog = False
    if not os.path.exists(train_data_dir+'cat/'):
        os.makedirs(train_data_dir+'cat/')
        move_cat = True
    if not os.path.exists(train_data_dir+'dog/'):
        os.makedirs(train_data_dir+'dog/')
        move_dog = True
    print("Split Dog and Cat")
    for file in tqdm(train_data_list):
        # file = train_data_list[1]
        if 'dog' in file and move_dog:
            shutil.move(train_data_dir+file, train_data_dir+'dog/'+file)
        elif 'cat' in file and move_cat:
            shutil.move(train_data_dir+file, train_data_dir+'cat/'+file)
        else:
            pass

    return 0


def split_data_into_validation():
    print("Create validation data")
    if not os.path.exists(validation_data_dir+'cat/'):
        os.makedirs(validation_data_dir+'cat/')
        cat_data_list = os.listdir(train_data_dir+'cat/')
        # Take random 20% of data
        cat_data_list = random.sample(cat_data_list, int(len(cat_data_list)*0.2))
        for file in cat_data_list:
            shutil.move(train_data_dir+'cat/'+file, validation_data_dir+'cat/'+file)

    if not os.path.exists(validation_data_dir+'dog/'):
        os.makedirs(validation_data_dir+'dog/')
        dog_data_list = os.listdir(train_data_dir+'dog/')
        # Take random 20% of data
        dog_data_list = random.sample(dog_data_list, int(len(dog_data_list)*0.2))
        for file in dog_data_list:
            shutil.move(train_data_dir+'dog/'+file, validation_data_dir+'dog/'+file)

    return True


def run_small_conv_net(model, train_generator, validation_generator, augmented):

    if not model:
        model = create_small_conv_net()

    if augmented:
        print("Training model with data augmentation")

    model.fit_generator(train_generator, validation_data=validation_generator, validation_steps=validation_samples, epochs=epoch, steps_per_epoch=train_samples)

    if save_weight:
        if augmented:
            model.save_weights('./small_conv_net_'+str(img_size)+'augmented.h5')
        else:
            model.save_weights('./small_conv_net_'+str(img_size)+'.h5')

    # computing loss and accuracy
    evaluation = model.evaluate_generator(validation_generator, nb_validation_samples)
    if augmented:
        print("Model evaluation with data augmentation {0}, {1}".format(evaluation[0], evaluation[1]))
    else:
        print("Model evaluation {0}, {1}".format(evaluation[0], evaluation[1]))

    return model


def run_vgg_16(model, train_generator, validation_generator, augmented):

    if not model:
        model = create_vgg_16_conv_net()

    if augmented:
        print("Training vgg model with data augmentation")

    model.fit_generator(train_generator, validation_data=validation_generator, validation_steps=validation_samples, epochs=epoch, steps_per_epoch=train_samples)

    if save_weight:
        if augmented:
            model.save_weights('./vgg_conv_net_'+str(img_size)+'augmented.h5')
        else:
            model.save_weights('./vgg_conv_net_'+str(img_size)+'.h5')

    # computing loss and accuracy
    evaluation = model.evaluate_generator(validation_generator, nb_validation_samples)
    if augmented:
        print("Model vgg16 evaluation with data augmentation {0}, {1}".format(evaluation[0], evaluation[1]))
    else:
        print("Model vgg16 evaluation {0}, {1}".format(evaluation[0], evaluation[1]))


    return model


def run_classifier(arg):

    split_data_into_cat_dog()

    ret = split_data_into_validation()

    if ret:
        # used to rescale the pixel values from [0, 255] to [0, 1] interval
        datagen = ImageDataGenerator(rescale=1./255)
        # automagically retrieve images and their classes for train and validation sets
        train_generator = datagen.flow_from_directory(train_data_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary')
        validation_generator = datagen.flow_from_directory(validation_data_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary')

        # Data augmentation for improving the model
        datagen_augmented = ImageDataGenerator(
                                    rescale=1./255,        # normalize pixel values to [0,1]
                                    shear_range=0.2,       # randomly applies shearing transformation
                                    zoom_range=0.2,        # randomly applies shearing transformation
                                    horizontal_flip=True)  # randomly flip the images

        train_generator_augmented = datagen_augmented.flow_from_directory(train_data_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary')

        if arg:
            print("Running Vgg convnet")
            model_vgg = run_vgg_16(None, train_generator, validation_generator, False)
            model_vgg = run_vgg_16(model_vgg, train_generator_augmented, validation_generator, True)
        else:
            print("Running small convnet")
            model = run_small_conv_net(None, train_generator, validation_generator, False)
            model = run_small_conv_net(model, train_generator_augmented, validation_generator, True)
    else:
        print("Dataset Split Error, Please check!")

    return 0


def main():

    if len(sys.argv)>1:
        arg = sys.argv[1]
        if arg=='1' or arg=='0':
            arg = int(arg)
            run_classifier(arg)
        else:
            print("Invalid argumet encountered")
    else:
        print("To run small convnet pass 0 as argument.")
        print("To run vgg16 pass 1 as argument.")

    return 0



if __name__ == '__main__':
    main()
