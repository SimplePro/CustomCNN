from keras.models import Sequential, load_model
import numpy as np
import json


class CustomCnn:

    def __init__(self, generator=True, image_shape=None, batch_size=None, model_name=None):
        if image_shape is None:
            raise Exception("image_shape must not be None")
        if batch_size is None:
            raise Exception("batch_size must not be None")
        if model_name is None:
            raise Exception("model_name must not be None")
        self.generator = generator
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.model_name = model_name
        self.class_indices = {}

    def _modeling(self, input_shape=None, model_info=None):
        if input_shape is None:
            raise Exception("input_shape must not be None")
        if model_info is None:
            raise Exception("model_info must not be None")

        self.model = Sequential()

        for mi in model_info:
            self.model.add(mi)

    def _model_compiling(self, loss=None, optimizer=None, metrics=None):
        if loss is None:
            raise Exception("loss must not be None")
        if optimizer is None:
            raise Exception("optimizer must not be None")
        if metrics is None:
            raise Exception("metrics must not be None")

        self.model.compile(
            loss=loss, optimizer=optimizer, metrics=metrics
        )

    def _summary(self):
        return self.model.summary()

    def _fit(self, train_data_generator, test_data_generator, train_directory, test_directory, train_set, test_set,
             epochs=None, steps_per_epoch=None, validation_steps=None, batch_size=None, verbose=None):
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

    def _evaluate(self, x_data, y_data):
        if self.generator:
            return self.model.evaluate_generator(self.test_set)
        if not self.generator:
            return self.model.evaluate(x_data, y_data)

    def _predict(self, img):
        if self.generator:
            result = self.model.predict(img)
            label = self.class_indices[np.argmax(result[0])]
            return label

        if not self.generator:
            label = self.model.predict_classes(img)
            return label

    def _save_model(self, directory=None):
        if directory is None:
            raise Exception("directory must not be None")
        self.model.save(directory + "/" + self.model_name + ".h5")

        data = {"generator": self.generator, "image_shape": self.image_shape, "batch_size": self.batch_size,
                "model_name": self.model_name, "class_indices": self.class_indices,
                "model": directory + "/" + self.model_name + ".h5"}
        try:
            data["train_set"] = self.train_set
            data["test_set"] = self.test_set
        except: pass

        try:
            data["x_train"] = self.x_train
            data["y_train"] = self.y_train
            data["x_test"] = self.x_test
            data["y_test"] = self.y_test
        except: pass

        with open(directory + "/" + self.model_name + ".json", "w") as f:
            json.dump(data, f, indent=4)

    def _load_model(self, directory=None, model_name=None):
        if directory is None:
            raise Exception("directory must not be None")
        if model_name is None:
            raise Exception("model_name must not be None")

        with open(directory + "/" + self.model_name + ".json", "r") as f:
            json_data = json.load(f)
            self.generator = json_data["generator"]
            self.image_shape = json_data["image_shape"]
            self.batch_size = json_data["batch_size"]
            self.model_name = json_data["model_name"]
            self.class_indices = json_data["class_indices"]
            self.model = load_model(json_data["model"])

            if "train_set" in json_data.keys():
                self.train_set = json_data["train_set"]
                self.test_set = json_data["test_set"]

            elif "x_train" in json_data.keys():
                self.x_train = json_data["x_train"]
                self.y_train = json_data["y_train"]
                self.x_test = json_data["x_test"]
                self.y_test = json_data["y_test"]
