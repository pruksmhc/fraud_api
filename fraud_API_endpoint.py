"""
Brizi API for fraud detection.
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
    return "It works!"

# Here, pass API the information to predict
@app.route('/fraud/', methods=['POST'])
def new_one():
	print(request.json)
	seats = request.json["seats"]
	fbp = request.json['fbpost']
	snapsh = request.json['snaps']
	lastSRS = request.json['lastSRS']
	ind_seats = request.json["ind_seats"]
	res = app.model.predict_banned(seats, fbp, snapsh, lastSRS, ind_seats)
	return jsonify({"result": ''})
	
# Paritlaf fit, retrain model
@app.route('/partial_fit/', methods=['POST'])
def partial_fit():
	fbp = int(request.args["fbpost"])
	snaps = int(request.args["snaps"])
	seats = request.args["seats"]
	lastSS = int(request.args["lastSRS"])
	res = app.model.partial_fit(X,Y)
	return jsonify({"result": res})




if __name__ == '__main__':
    app.run(debug=True)