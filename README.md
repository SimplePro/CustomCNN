# CustomCNN
------------------
CustomCnn function
-----------
1. auto modeling
2. modeling
3. compiling
4. summary
5. set data generator
6. fit (ImageDataGenerator version, Generally version)
7. evaluate (ImageDataGenerator version, Generally version)
8. predict (ImageDataGenerator version, Generally version)
9. save_model
10. load_model
11. return info

CnnSet function
----------------
1. add model
2. save CnnSet
3. load CnnSet
4. return one model in CnnSet
5. delete one model in CnnSet
6. delete CnnSet
7. return info

CustomCnn example
------------
``` python
from CustomModel import CustomCnn
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

custom_cnn = CustomCnn(generator=False, model_name="mnist_test1")
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255.

batch_size = 128
class_n = 10
epochs = 3

y_train = keras.utils.to_categorical(y_train, class_n)
y_test = keras.utils.to_categorical(y_test, class_n)

custom_cnn._modeling(model_info=[
                        Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu", input_shape=input_shape),
                        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                        Conv2D(64, (2, 2), activation="relu", padding="same"),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.25),
                        Flatten(),
                        Dense(1000, activation="relu"),
                        Dropout(0.5),
                        Dense(class_n, activation="softmax")
                     ])

custom_cnn._summary()

custom_cnn._compiling(loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

custom_cnn._fit(train_set=(x_train, y_train), test_set=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)
score = custom_cnn._evaluate(x_data=x_test, y_data=y_test)
print(score)

custom_cnn._save_model("./test/")
```

CnnSet example
-----------
``` python
from ModelsSet import CnnSet

cnn_set = CnnSet(cnn_set_name = "cnn_set1")
cnn_set._add_model(custom_cnn)  # add model
cnn_set_pred = cnn_set._get("mnist_test1")._predict(img)  # get and predict
print(pred)
## 3

cnn_set._save_cnn_set("./")  # save cnn_set


cnn_set._load_cnn_set(cnn_set_path = "./cnn_set1")  # load cnn_set

cnn_set._info()  # info

cnn_set._delete_model("mnist_test1") # delete model

cnn_set._delete_cnn_set()  # delete cnn_set
```
