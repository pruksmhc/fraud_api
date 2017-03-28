
__author__ = "Yada Pruksachatkan"

import requests


def test_api():
	url = "http"
	seats = ["BMW Suite-1-1","100-A-1","100-A-1","SB06-1-1","SB06-1-1","100-PP-10","100-O-10","100-O-1","SB06-1-1"]
	url = "http://127.0.0.1:5000/fraud/"
	ind_seats = {"BMW Suite-1-1": 0, "100-A-1": 0,"SB06-1-1":0, "100-PP-10":0, "100-O-10":0 }
	data = {'fbpost': 2, 'snaps' : 10, 'lastSRS': 10, 'seats': seats, 'ind_seats':ind_seats}
	r = requests.post(url, json=data, allow_redirects=True)
	print(r)


if __name__ == "__main__":
	test_api()

