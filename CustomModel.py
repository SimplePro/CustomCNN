from keras.models import Sequential, load_model
import numpy as np
import json

from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing import image

from Preprocessing import ImagePreprocess

from keras_preprocessing.image import ImageDataGenerator

import os


class BaseModeling:

    # auto modeling
    def _auto_modeling(self, input_shape=None, class_n=None):
        if input_shape is None:
            raise Exception("input_shape must not be None")
        if class_n is None:
            raise Exception("class_n must not be None")

        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(class_n, activation='softmax'))

    # modeling
    def _modeling(self, model_info=None):
        if model_info is None:
            raise Exception("model_info must not be None")

        self.model = Sequential()

        for mi in model_info:
            self.model.add(mi)

    # compiling
    def _compiling(self, loss=None, optimizer=None, metrics=None):
        if loss is None:
            raise Exception("loss must not be None")
        if optimizer is None:
            raise Exception("optimizer must not be None")
        if metrics is None:
            raise Exception("metrics must not be None")

        self.model.compile(
            loss=loss, optimizer=optimizer, metrics=metrics
        )

    # summary
    def _summary(self):
        return self.model.summary()


class CustomCnn(BaseModeling):

    # init
    def __init__(self, generator=False, model_name=None):
        if model_name is None:
            raise Exception("model_name must not be None")

        self.generator = generator
        self.model_name = model_name
        self.class_indices = {}

    # method to set train_set and test_set again
    def _set_data_generator(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    # fit
    def _fit(self, train_data_generator=None, test_data_generator=None, train_directory=None, test_directory=None,
             train_set=None, test_set=None, target_size=None, epochs=None, steps_per_epoch=None, validation_steps=None,
             batch_size=None, verbose=None):
        if batch_size is None:
            raise Exception("batch_size must not be None")

        self.image_shape = target_size
        self.batch_size = batch_size

        if self.generator:
            if train_data_generator is None:
                raise Exception("train_data_generator must not be None")
            if test_data_generator is None:
                raise Exception("test_data_generator must not be None")
            if steps_per_epoch is None:
                raise Exception("steps_per_epoch must not be None")
            if epochs is None:
                raise Exception("epochs must not be None")
            if validation_steps is None:
                raise Exception("validation_steps must not be None")
            if target_size is None:
                raise Exception("target_size must not be None")

            self.train_set = train_data_generator.flow_from_directory(
                train_directory,
                batch_size=self.batch_size,
                target_size=self.image_shape,
                class_mode='categorical'
            )

            self.test_set = test_data_generator.flow_from_directory(
                test_directory,
                batch_size=self.batch_size,
                target_size=self.image_shape,
                class_mode="categorical"
            )

            self.class_indices = self.test_set.class_indices

            self.model.fit_generator(
                self.train_set,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=self.test_set,
                validation_steps=validation_steps
            )

        if not self.generator:
            if train_set is None:
                raise Exception("train_set must not be None")
            if test_set is None:
                raise Exception("test_set must not be None")
            if verbose is None:
                raise Exception("verbose must not be None")
            if epochs is None:
                raise Exception("epochs must not be None")
            if batch_size is None:
                raise Exception("batch_size must not be None")

            self.x_train, self.y_train = train_set
            self.x_test, self.y_test = test_set

            self.model.fit(self.x_train, self.y_train, validation_data=test_set,
                           epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    def _evaluate(self, x_data=None, y_data=None):
        if self.generator:
            evaluate_score = self.model.evaluate_generator(self.test_set)
            return evaluate_score

        if not self.generator:
            evaluate_score = self.model.evaluate(x_data, y_data)
            return evaluate_score

    # predict
    def _predict(self, x):

        if self.generator:
            result = self.model.predict(x)
            label = ""
            items = self.class_indices.items()
            for key, value in items:
                if value == np.argmax(result[0]):
                    label = key

            return label

        if not self.generator:
            label = self.model.predict_classes(x)
            return label

    # predict img
    def _predict_img(self, img=None, path=None):
        if img is None and path is None:
            raise Exception("img and path must not be None")

        result = 0

        try:
            img = image.load_img(path)
        except:
            raise Exception("the image file could not be found in the specified path.")

        if not img is None:
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            result = self._predict(img)

        return result

    # method to save model as file
    def _save_model(self, directory=None):
        if directory is None:
            raise Exception("directory must not be None")

        self.model.save(directory + self.model_name + ".h5")

        data = {"generator": self.generator, "image_shape": self.image_shape, "batch_size": self.batch_size,
                "model_name": self.model_name, "class_indices": self.class_indices,
                "model": os.path.abspath(directory + self.model_name + ".h5")}

        with open(directory + self.model_name + ".json", "w") as f:
            json.dump(data, f)

    # method to load model as model_name
    def _load_model(self, directory=None, model_name=None):
        if directory is None:
            raise Exception("directory must not be None")
        if model_name is None:
            raise Exception("model_name must not be None")

        with open(os.path.abspath(directory + model_name + ".json"), "r") as f:
            json_data = json.load(f)

        self.generator = json_data["generator"]
        self.image_shape = json_data["image_shape"]
        self.batch_size = json_data["batch_size"]
        self.model_name = json_data["model_name"]
        self.class_indices = json_data["class_indices"]
        self.model = load_model(json_data["model"])

    # method to return info
    def _info(self):
        return {"generator": self.generator, "image_shape": self.image_shape, "batch_size": self.batch_size,
                "model_name": self.model_name, "class_indices": self.class_indices}


class SimpleCnn(CustomCnn, ImagePreprocess):

    def __init__(self, generator=False, model_name=None):
        super().__init__(generator, model_name)

        self.trainDataGen = ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.1,
                                               height_shift_range=0.1,
                                               shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                               vertical_flip=True,
                                               fill_mode='nearest')

        self.testDataGen = ImageDataGenerator(rescale=1. / 255)

        self.image_shape = ()
        self.train_set = None
        self.test_set = None

    def set_dataset(self, train_set=None, test_set=None, train_directory=None, test_directory=None, dsize=None):
        if not self.generator:
            if train_set is None:
                raise Exception("train_set must not be None")
            if test_set is None:
                raise Exception("test_set must not be None")

            self.train_set = train_set
            self.test_set = test_set

        if self.generator:
            if dsize is None:
                raise Exception("dsize must not be None")
            self.resize_img(directory=train_directory, size=dsize)
            self.resize_img(directory=test_directory, size=dsize)

            self.image_shape = dsize

            self.train_set = self.trainDataGen.flow_from_directory(
                train_directory,
                batch_size=128,
                target_size=self.image_shape,
                class_mode='categorical'
            )

            self.test_set = self.testDataGen.flow_from_directory(
                test_directory,
                batch_size=128,
                target_size=self.image_shape,
                class_mode="categorical"
            )

            self.class_indices = self.test_set.class_indices

    def fit(self, class_n, input_shape=None, epochs=None):
        if epochs is None:
            raise Exception("epochs must not be None")
        if input_shape is None:
            raise Exception("input_shape must not be None")

        if self.generator:
            self._auto_modeling(input_shape=input_shape, class_n=class_n)
            self._compiling(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            self.model.fit_generator(
                self.train_set,
                steps_per_epoch=30,
                epochs=epochs,
                validation_data=self.test_set,
                validation_steps=10
            )

        if not self.generator:
            if input_shape is None:
                raise Exception("input_shape must not be None")

            self._auto_modeling(input_shape=input_shape, class_n=class_n)
            self._compiling(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            self.model.fit(self.train_set[0], self.train_set[1], validation_data=self.test_set,
                           epochs=epochs, batch_size=128, verbose=0)

    def predict(self, x=None, path=None):
        if x is None and path is None:
            raise Exception("x and path must not be None")
        if self.generator:
            if path is None:
                img = self.img_to_pred(x)
                pred = self._predict(img)
                return pred
            
            if not path is None:
                img = image.load_img(path)
                img = self.img_to_pred(img)
                pred = self._predict(img)
                return pred

        if not self.generator:
            if x.ndim == 4:
                pred = self.model.predict_classes(x)
            elif x.ndim == 3:
                pred_data = np.expand_dims(x, axis=0)
                pred = self.model.predict_classes(pred_data)
            return pred

