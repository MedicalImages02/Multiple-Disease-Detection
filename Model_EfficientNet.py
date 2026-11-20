from keras import Sequential
from keras.applications import EfficientNetB0
import numpy as np
from Evaluation import evaluation
from keras.models import Sequential
from keras.layers import Dense


def Model_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target):
    IMG_SIZE = 256

    Train_x = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_x[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_x = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_x[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    efficient_net = EfficientNetB0(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False,
        pooling='max'
    )

    model = Sequential()
    model.add(efficient_net)  # add EfficientNetB0 base model
    model.add(Dense(units=Train_Target.shape[1], activation='relu'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_x, Train_Target, epochs=100)
    pred = model.predict(Test_x)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)
    return Eval, pred

