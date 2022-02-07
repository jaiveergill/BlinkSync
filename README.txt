BlinkSync

The goal is to build a low-cost and efficient device to convert a patient’s blinks into words through Morse Code and machine learning so that the 
patient can communicate much more easily by using biometric sensors from Bitalino to convert blinks into words. 

src:
    stream:
        data: contain all of the data used in the project (training and testing for model along with data for trials.py)
	graph_model.py: used for Tensorboard and creating visualizations
	receive_stream.py: main file – connects to Bitalino and deciphers blinks live
	train_model.py: creates and trains model
	trials.py: Tests the models on 6 trials
     web:
	public: contains javascript, css, and images for the project
	views: HTML files (ejs used in this project)
	app.js: main NodeJS file used in the project
	data.txt: used when run locally
	package.json: NodeJS packages
	package-lock.json


