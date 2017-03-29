
__author__ = "Yada Pruksachatkan"

import requests


def test_api():
	seats = ["100-U-403","100-U-405","100-U-406","100-U-407"]
	url = "http://127.0.0.1:5000/fraud/"
	data = {'fbpost': 0, 'snaps' : 0, 'lastSRS': 4, 'seats': seats} # R6xIr0vs6v
	r = requests.post(url, json=data, allow_redirects=True)
	print("and the answer is")
	print(r.text)
	seats = ["100-F-345","100-H-196","100-A-1","100-A-200","100-A-230","100-A-200","100-A-200","100-A-230","100-A-260","100-A-260","100-A-266","100-A-268","100-A-268","100-D-270","100-D-271","100-E-273"]
	data = {'fbpost': 0, 'snaps' : 3, 'lastSRS': 11, 'seats': seats} # rGvb7FUiD8
	r = requests.post(url, json=data, allow_redirects=True)
	print("second is ")
	print(r.text)
	# SHOULD BE 1
	seats = ["100-R-360"] 
	data = {'fbpost': 0, 'snaps' : 3, 'lastSRS': 1, 'seats': seats} # ISwCU5kFbt
	r = requests.post(url, json=data, allow_redirects=True)
	print("third is")
	print(r.text)
	# SHOULD BE 0
	seats = ["100-KK-247","100-KK-247","100-KK-247","100-KK-247","100-KK-247"]
	data = {'fbpost': 0, 'snaps' : 1, 'lastSRS': 1, 'seats': seats} # ISwCU5kFbt
	r = requests.post(url, json=data, allow_redirects=True)
	print("fourth is")
	print(r.text)
	url = "http://127.0.0.1:5000/partial_fit/"
	data = {'fbpost': 0, 'snaps' : 1, 'lastSRS': 1, 'seats': seats, 'res': 1}
	r = requests.post(url, json=data, allow_redirects=True)
	print("now partial fit")


if __name__ == "__main__":
	test_api()

