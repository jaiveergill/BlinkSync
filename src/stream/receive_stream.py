# imports
from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
import scipy
import requests
import tensorflow.keras as keras
model_dir = "src/stream/saved_model"

model = keras.models.load_model(model_dir, compile=True) # load the model
print("Model loaded.")

# parameters
use_std = False # standard deviation or model
web = False # localhost or https

morse_dict = {
    '....' : 'h', '.-' : 'a', '-...' : 'b', '-.-.' : 'c', '-..' : 'd', '.' : 'e', '..-.' : 'f', '--.' : 'g', '..' : 'i', '.---' : 'j', '-.-' : 'k', '.-..' : 'l', '--' : 'm', '-.' : 'n', '---' : 'o', '.--.' : 'p', '--.-' : 'q', '.-.' : 'r', '...' : 's', '-' : 't', '..-' : 'u', '...-' : 'v', '.--' : 'w', '-..-' : 'x', '-.--' : 'y', '--..' : 'z', '.-.-.-' : '.', '..--..' : '?', '--..--' : ',', '/' : ' '
}

# Resolve an available OpenSignals stream
print("# Looking for an available OpenSignals stream...")
os_stream = resolve_stream("name", "OpenSignals")
# Create an inlet to receive signal samples from the stream
inlet = StreamInlet(os_stream[0])

data = [] 
counter = 0
waving = False


silence1 = 0.0 # amount of time since last letter
silence2 = 0.0 # amount of time since last space
full_sequence = "" # the full message currently displayed
waiting = 0 # cooldown between blink guesses
current_sequence = "" # current morse code sequence
counter = 0
while True:
    try:
        if 2.9 < silence1 < 3.1:
            """
            This code converts the current morse code sequence and makes it into a letter
            """
            try:
                letter = morse_dict[current_sequence]
            except KeyError:
                print(f"{current_sequence} is not a morse code letter")
                silence1 = 0
                silence2 = 0
                current_seq = ''
                continue
            full_sequence += letter
            current_sequence = ''
            silence1 = 0
        if 4.9 < silence2 < 5.1:
            """
            This adds a space to the full sequence
            """
            print(silence2)
            full_sequence += ' '
            current_seq = ''
            silence1 = 0
            silence2 = 0
        sample, timestamp = inlet.pull_sample()
        data.append(int(sample[1]))
        
        if waiting > 0:
            """
            Doesn't run the next code if in the cooldown
            """
            counter += 1
            waiting -= 1
            continue

        if counter % 50 == 0 and counter > 1000:
            if counter % 100 == 0:
                if web: # post to a website if http, and edit file if localhost
                    r = requests.post("https://blinksync.herokuapp.com/data", data={"text":f"Current: {current_sequence}, Full: {full_sequence}, silence1: {silence1}, Silence2: {silence2}"})
                else:
                    with open("src/web/data.txt", "w") as file:
                        file.write(f"Morse: {current_sequence}, Word: {full_sequence}, Until letter: {silence1}, Until space: {silence2}")
                    
                print(f"Current: {current_sequence}, Full: {full_sequence}, silence1: {silence1}, Silence2: {silence2}")
            wave = data[counter-1000:counter] # the wave to be sent to the model
            wave = scipy.signal.savgol_filter(wave, 101, 3) # Savitzky-Golay filter
            
            if np.std(wave) > 15: # just an initial filter for optimization reasons
                pred = list(model.predict(np.array([wave]))[0]) # model's prediction
                pred = pred.index(max(pred))
                if pred != 2:
                    print(pred)
                    current_sequence += [".", "-"][pred]
                    silence1 = 0
                    silence2 = 0
                    waiting = 1250
                else:
                    silence1 += 0.05
                    silence2 += 0.05
            else:
                silence1 += 0.05
                silence2 += 0.05
                

        counter += 1
    except Exception as e:
        print(e)
        
        