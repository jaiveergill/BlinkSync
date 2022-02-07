global counter # global counter variable

# imports
from gc import callbacks
import tensorboard
import datetime
counter = 0
from scipy.interpolate import interp1d
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPooling1D, Flatten # all of the layers for the neural network
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy
import numpy as np
import random

# training data with Savitzky-Golay filter applied
left = list(nk.read_bitalino("src/stream/data/left_eye_100_blinks_converted.txt")[0]["EEGBITREV"])
right = list(nk.read_bitalino("src/stream/data/right_eye_100_blinks_converted.txt")[0]["EEGBITREV"])
left = scipy.signal.savgol_filter(left, 101, 3)
right = scipy.signal.savgol_filter(right, 101, 3)
counter = 0
waiting = 0
data = []
lefts = []

while counter < len(left):
    """
    Separates the left blink waves out into an array using brute force standard deviation
    """
    while waiting:
        counter += 1
        waiting -= 1
    if counter % 50 == 0 and counter > 1000:
        trend = np.std(left[counter-1000:counter])
        if trend > 23:
            wave = left[counter-1000:counter]
            lefts.append(wave)
            waiting = 975

    counter += 1


print(len(lefts))
counter = 0
waiting = 0
data = []
rights = []

while counter < len(right):
    """
    Separates the right blink waves out into an array using brute force standard deviation
    """
    while waiting:
        counter += 1
        waiting -= 1
    if counter % 50 == 0 and counter > 1000:
        trend = np.std(right[counter-1000:counter])
        if trend > 25:
            wave = right[counter-1000:counter]
            rights.append(wave)
            waiting = 975

    counter += 1

print(len(rights))


resting = list(nk.read_bitalino("src/stream/data/resting_converted.txt")[0]["EEGBITREV"]) # resting test data
resting = scipy.signal.savgol_filter(resting, 101, 3)
resting = resting[1000:len(resting)//3+1000] # take only first 1/3 of data to match with amount of left and right waves
x = len(resting)//1000 
resting = resting[:1000*x] # split into slices of 1000
resting = list(np.array_split(resting, x))
print(len(resting))



def trains(x_tt, y_tt):
    """
    Converts splits x and y arrays into trains and tests
    """
    x_train = x_tt[0:round(len(x_tt)*4/5)]
    x_test = x_tt[round(len(x_tt)*4/5)+1:len(x_tt)-1]
    y_train = y_tt[0:round(len(y_tt)*4/5)]
    y_test = y_tt[round(len(y_tt)*4/5)+1:len(y_tt)-1]
    return x_train, x_test, y_train, y_test
def s3(x, y, z):
    xyz = np.concatenate([x, y, z])
    y_tt = [None for _ in range(len(xyz))]
    x_tt = [None for _ in range(len(xyz))]
    arr = [i for i in range(len(xyz))]
    counter = 0
    while arr != []:
        rand = random.randrange(len(arr))
        arr = arr.remove(arr[rand])
        x_tt[rand] = xyz[counter]
        if counter < len(x):
            y_tt[rand] = 0
        elif counter > len(y):
            y_tt[rand] = 2
        else:
            y_tt[rand] = 1
        x_train, x_test, y_train, y_test = trains(x_tt, y_tt)
        counter += 1

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    
"""
The following code shuffles the left blink, right blink, and resting data.
It puts them into two big x and y arrays for the model to learn from.
"""
full_arr = np.concatenate([rights, lefts, resting])
x_tt = [None for i in range(len(full_arr))]
y_tt = [None for i in range(len(full_arr))]
arr = [i for i in range(len(full_arr))]
counter = 0
done = []
while counter < len(full_arr):
    rand = random.randrange(len(full_arr)) 
    if rand in done:
        continue
    done.append(rand)
    x_tt[rand] = full_arr[counter]
    if counter < len(rights):
        y_tt[rand] = 0
    elif counter > len(rights)+len(lefts):
        y_tt[rand] = 2
    else:
        y_tt[rand] = 1
    counter += 1
x_train, x_test, y_train, y_test = trains(x_tt, y_tt)

def ml(x_train, y_train, x_test, y_test, show_graph=False, log=False, save=False):
    """
    Creates the model and tests it
    Args:
        x_train
        y_train
        x_test
        y_test
        
    Used to train and test the model
    
    Kwargs:
        show_graph: when set to True it shows epoch_loss and epoch_accuracy graphs
        log: when set to True creates Tensorboard log
        save: saves model to a folder when set to True
    """
    model = keras.Sequential()
    model.add(Conv1D(filters=128, kernel_size=64, activation='relu', input_shape=(1000, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'],
                )
    if log:
        log_dir = "src/stream/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(x_train, y_train, epochs=20, batch_size=10, callbacks=[tensorboard_callback])
    else:
        history = model.fit(x_train, y_train, epochs=20, batch_size=10)

    model.evaluate(x_test, y_test, verbose=2)

    if save:
        model.save('src/stream/saved_model')
    if show_graph:
        acc = history.history['accuracy']
        x = np.array([i+1 for i in range(20)])
        cubic_interpolation_model = interp1d(x, acc, kind = "cubic")
        X_=np.linspace(x.min(), x.max(), 500)
        Y_=cubic_interpolation_model(X_)
        
        
        loss = history.history['loss']
        x2 = np.array([i+1 for i in range(20)])
        cubic_interpolation_model2 = interp1d(x2, loss, kind = "cubic")
        X_2=np.linspace(x.min(), x.max(), 500)
        Y_2=cubic_interpolation_model2(X_2)
        
        plt.plot(X_, Y_)
        plt.plot(X_2, Y_2)
        plt.title("Accuracy (blue) and loss (orange) of the model during training")
        plt.xlabel("epoch")
        plt.ylabel("accuracy/loss")
        plt.show()

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
ml(x_train, y_train, x_test, y_test, save=True)