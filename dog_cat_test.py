from keras.preprocessing.image import ImageDataGenerator
from CustomModel import CustomCnn
from keras import layers

trainDataGen = ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                                  fill_mode='nearest')

testDataGen = ImageDataGenerator(rescale=1. / 255)

custom_cnn = CustomCnn(generator=True, model_name="dog_cat_test2")

custom_cnn._modeling([
    layers.Conv2D(16, (3, 3), (1, 1), padding="same", input_shape=(64, 64, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(rate=0.3),
    layers.Conv2D(32, (3, 3), (1, 1), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(rate=0.3),
    layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(rate=0.3),
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(2, activation="sigmoid")
])
custom_cnn._compiling(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"]
                      )

custom_cnn._fit(train_data_generator=trainDataGen, test_data_generator=testDataGen, steps_per_epoch=30,
                train_directory="./dog_cat_dataset/training_set/", test_directory="./dog_cat_dataset/test_set/",
                epochs=100, batch_size=20, verbose=0, target_size=(64, 64), validation_steps=10)

score = custom_cnn._evaluate()
print(score)
print(custom_cnn.class_indices)

custom_cnn._save_model("./test/")
