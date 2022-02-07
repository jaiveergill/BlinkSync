global counter

# imports
import numpy as np
import scipy
import neurokit2 as nk
import matplotlib.pyplot as plt
import tensorflow.keras as keras

model_dir = 'src/stream/saved_model'

# all of the files full of testing data being loaded in
left1 = list(nk.read_bitalino("src/stream/data/testing/left_testing1.txt")[0]["EEGBITREV"])
left2 = list(nk.read_bitalino("src/stream/data/testing/left_testing2.txt")[0]["EEGBITREV"])
left3 = list(nk.read_bitalino("src/stream/data/testing/left_testing3.txt")[0]["EEGBITREV"])
right1 = list(nk.read_bitalino("src/stream/data/testing/right_testing1.txt")[0]["EEGBITREV"])
right2 = list(nk.read_bitalino("src/stream/data/testing/right_testing2.txt")[0]["EEGBITREV"])
right3 = list(nk.read_bitalino("src/stream/data/testing/right_testing3.txt")[0]["EEGBITREV"])
all_files = [left1, left2, left3, right1, right2, right3]
print("Files loaded")

model = keras.models.load_model(model_dir, compile=True) # load the model
print("Model loaded.")

total_score = 0 # total number of correctly guessed waves
total_len = 0 # total number of guesses

for i in range(6):
    "For each array, it the model makes its guesses on the amount of blinks in the file"
    arr = all_files[i]
    guesses = []
    data = []    
    waiting = 0
    counter = 0
    waves = 0
    while counter < len(arr):
        
        data.append(arr[counter])
        if waiting:
            waiting -= 1
            counter += 1
            continue
        if counter % 50 == 0 and counter > 1000:
            wave = data[counter-1000:counter]
            wave = scipy.signal.savgol_filter(wave, 101, 3)
            if np.std(wave) > 20:
                pred = list(model.predict(np.array([wave]))[0])
                pred = pred.index(max(pred))
                if pred != 2:
                    guesses.append(pred)
                    waiting = 1250
        counter += 1
    num = 0
    if i <= 2: # the first 3 files are left blinks, the next three are right blinks
        num = 1
    score = 0
    for guess in guesses:
        if guess == num:
            score += 1
    total_score += score # adds to total score
    total_len += len(guesses) # adds to total guesses
    print(score/len(guesses)) # prints out results for this trial

print(total_score/total_len) # total results (average of all 6 trials)
