
__author__ = "Yada Pruksachatkan"

import requests
import json

# Fraud_api.py. he udnerstood. 
# I need to sleep, I drank coffee and it doens't help. It' ust feels so good to be sick. To not have any expectations.  
# To have no expetations of you.  - Yada found this out now. For the ARIMA models.  Yada Pruksachaktun had this here. 

def test_api():
	seats = ["100-U","100-U","100-U","100-U"]
	url = "http://127.0.0.1:5000/fraud/"
	data = {'fbpost': 0, 'snaps' : 0, 'lastSRS': 4, 'seats': seats, 'uniqueFB': 0, 'uniqueSnaps': 0} # R6xIr0vs6v
	r = requests.post(url, json=data, allow_redirects=True)
	print("and the answer is")
	print(r.text)
	seats = ["100-F","100-H","100-A","100-A", "100-D","100-E"]
	data = {'fbpost': 0, 'snaps' : 3, 'lastSRS': 11, 'seats': seats, 'uniqueFB': 0, 'uniqueSnaps': 3} # rGvb7FUiD8
	r = requests.post(url, json=data, allow_redirects=True)
	print("second is ")
	print(r.text)
	# SHOULD BE 1
	seats = ["100-R"] 
	data = {'fbpost': 0, 'snaps' : 3, 'lastSRS': 1, 'seats': seats, 'uniqueFB': 0, 'uniqueSnaps': 3} # ISwCU5kFbt
	r = requests.post(url, json=data, allow_redirects=True)
	print("third is")
	print(r.text)
	while (json.loads(r.text)["result"] == 0.0):
		url = "http://127.0.0.1:5000/partial_fit/"
		data = {'fbpost': 0, 'snaps' : 3, 'lastSRS': 1, 'seats': seats, 'res' : 1, 'uniqueFB': 0, 'uniqueSnaps': 3} # ISwCU5kFbt
		r = requests.post(url, json=data, allow_redirects=True)
		print("now partial fit")
		url = "http://127.0.0.1:5000/fraud/"
		data = {'fbpost': 0, 'snaps' : 3, 'lastSRS': 1, 'seats': seats, 'uniqueFB': 0, 'uniqueSnaps': 3} # ISwCU5kFbt
		r = requests.post(url, json=data, allow_redirects=True)

	# SHOULD BE 0
	url = "http://127.0.0.1:5000/fraud/"
	seats = ["100-KK","100-KK","100-KK", "100-KK", "100-KK"]
	data = {'fbpost': 5, 'snaps' : 10, 'lastSRS': 1, 'seats': seats, 'uniqueFB': 2, 'uniqueSnaps': 5} # ISwCU5kFbt
	r = requests.post(url, json=data, allow_redirects=True)

	while (json.loads(r.text)["result"] == 1.0):
		url = "http://127.0.0.1:5000/partial_fit/"
		data = {'fbpost': 5, 'snaps' : 10, 'lastSRS': 1, 'seats': seats, 'res': 0, 'uniqueFB': 2, 'uniqueSnaps': 5} # ISwCU5kFbt
		r = requests.post(url, json=data, allow_redirects=True)
		url = "http://127.0.0.1:5000/fraud/"
		data = {'fbpost': 5, 'snaps' : 10, 'lastSRS': 1, 'seats': seats, 'uniqueFB': 2, 'uniqueSnaps': 5} # ISwCU5kFbt
		r = requests.post(url, json=data, allow_redirects=True)
	print("fourth is")
	print(r.text)
	#  after running htis a few times,should change the second to last to predict 1.0


if __name__ == "__main__":
	test_api()

