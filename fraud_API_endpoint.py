"""
Brizi API for fraud detection.
"""
__author__ = "Yada Pruksachatkan"

from base64 import urlsafe_b64decode
from collections import defaultdict
from cStringIO import StringIO
from flask import Flask, jsonify, request
import os
from signal import signal, SIGPIPE, SIG_DFL
from text_extraction import get_text
from time import time
from fraud import FraudDetection
from werkzeug.utils import secure_filename
try:
    import Image
except ImportError:
    from PIL import Image

signal(SIGPIPE,SIG_DFL)
app = Flask(__name__)

@app.route('/fraud/', methods=['POST']):
	# Do all the preprocessing
	model = FraudDetection()
	return model.predict_banned(user_id, fbpost, snaps, seats, lastSRS)
