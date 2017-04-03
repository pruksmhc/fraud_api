"""
API for fraud detection.
"""
__author__ = "Yada Pruksachatkan"

from base64 import urlsafe_b64decode
from collections import defaultdict
from flask import Flask, jsonify, request
import os
import json
from signal import signal, SIGPIPE, SIG_DFL
from fraud import FraudDetection
from werkzeug.utils import secure_filename
from flask import request
try:
    import Image
except ImportError:
    from PIL import Image

signal(SIGPIPE,SIG_DFL)
app = Flask(__name__, template_folder='.')
app.model = FraudDetection()

@app.route('/')
def index():
    return

@app.route('/fraud/', methods=['POST'])
def new_one():
	"""
	The point of this is to be the API endpoint for checking if the user 
	is banned or not.
	For each slugDate
	The parameters are: 
	Seats -> SRSString, distribution of seats in SRSString, disregard the seat number. 
	FBP - Facebook posts (aggregated per user) for a certain date
	snapsh - Number of snapshots (aggreagted per user for a certain date)
	LastSRSCount 
	uniqueFacebook.-> unique facebook
	uniqueSnaps -> unique snapshots & fbp (just try this)
	"""
	seats = request.json["seats"]
	fbp = request.json['fbpost']
	snapsh = request.json['snaps']
	lastSRS = request.json['lastSRS']
	unique_fbp = request.json['uniqueFB']
	unique_snaps = request.json['uniqueSnaps']
	res = app.model.predict_banned(seats, fbp, snapsh, lastSRS, unique_fbp, unique_snaps)
	return jsonify({"result": res})

# Partial fit, retrain model
@app.route('/partial_fit/', methods=['POST'])
def partial_fit():
	"""
	The point of this is to be the API endpoint for training the model in 
	real time.
	The parameters are: 
	Seats -> SRSString
	FBP - Facebook posts (aggregated per user) for a certain date
	snapsh - Number of snapshots (aggreagted per user) for a certain date
	LastSRSCount
	res - the human-generated label of whether the user was banned or not
	"""
	seats = request.json["seats"]
	fbp = request.json['fbpost']
	snapsh = request.json['snaps']
	lastSRS = request.json['lastSRS']
	res = request.json["res"]
	unique_fbp = request.json['uniqueFB']
	unique_snaps = request.json['uniqueSnaps']
	app.model.partial_train(seats, fbp, snapsh, lastSRS, res, unique_fbp, unique_snaps)
	return jsonify({"result": ''})




if __name__ == '__main__':
    app.run(debug=True)