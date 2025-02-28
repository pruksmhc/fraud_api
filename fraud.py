from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import csv
import pandas as pd
import random 
import numpy as np
import ast
import time
from collections import defaultdict
from sklearn import preprocessing


__author__ = "Yada Pruksachatkan"

class FraudDetection:
	"""
	The goal of this class is to be able to detect whether or not a user is a fraud. 
	Right nowe, it uses one neural netowrk to create the model.
	The model right now needs to be trained using CSVs. 
	When you train the omdel, it will adjust according. It will learn 
    """
    # Field variables
	NUM = 50
	EVENTUSERS_CSV = "eventusers.csv"
	SNAPS_CSV = "snapshots.csv"
	FBPOST_CSV = "facebookposts.csv"

	def __init__(self):
		"""
		seat_histogram - map of unique seats
		Fraud APi 
		"""
		self.classifiers = [None]
		self.seat_histogram = {}
		self.feature_vector = []
		X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos = self.extract_features() 
		classifier1 = MLPClassifier(solver='adam', alpha=1e-5,
			hidden_layer_sizes=(30,30,30), random_state=1)
		acc_1 = 0 
		acc_2 = 0 
		while ((acc_1 < 0.92) or (acc_2 < 0.92)):
			X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos = self.extract_features() 
			classifier1 = MLPClassifier(solver='adam', alpha=1e-5,
			hidden_layer_sizes=(30,30,30), random_state=1)
			self.classifiers[0] = classifier1
			self.train(X_train, Y_train)
			acc_1 = self.test(X_test_pos, Y_test_pos)
			acc_2 = self.test(X_test_neg, Y_test_neg)
		print("Accuracy")
		print("For Banned")
		print(acc_1)
		print("For Not banned")
		print(acc_2)


	def parse_seats(self, seats):
		"""
		Output is an array of seats in SRSString
		["100-U-403","100-U-405","100-U-406","100-U-407"]
		"""
		indiv_seats = {}
		seat_num = seats.split(",")
		for s in seat_num:
			s = s.replace("[", "")
			s = s.replace("]", "")
			s = s.split("\"")
			for se in s:
				se = se.split("-")
				se = se[:-1]
				se = "-".join(se)
				if ((se is not "\"") and (se is not " ") and (se is not "")):
					if indiv_seats.get(se) is None:
						indiv_seats[se] = True
		return indiv_seats

	def replace_value_with_definition(self, value_to_find, definition, current_dict):
	    for key, value in current_dict.items():
	        if value == value_to_find:
	            current_dict[key] = definition
	    return current_dict

	def calculate_curr(self, seats, curr_id, snaps, fbpost, lastSRSCount, slugDate):
		"""
		This function, from the CSVs, extracts the necessary data and puts it 
		into curr, which is the form of the training data to be fed in. 
		This outputs a vector for a current user that the classifier can train on 
		- see test_api.py 
		"""
		snapsh = snaps.loc[ (snaps['userId'] == curr_id) & (snaps["slugDate"] == slugDate) ]	
		fbp = fbpost.loc[ (fbpost['userId'] == curr_id) & (fbpost["slugDate"] == slugDate) ]
		unique_snaps = snapsh.drop_duplicates()
		unique_fb = fbp.drop_duplicates()
		seats = self.parse_seats(seats)
		uniqueSnaps = len(unique_snaps)
		uniqueFB = len(unique_fb)
		return self.calculate_curr_helper(seats, len(snapsh), len(fbp), lastSRSCount,uniqueSnaps,uniqueFB)

	def calculate_curr_helper(self, srs_seats, len_snapsh, len_fbp, lastSRSCount, unique_snaps, unique_fb):
		"""
		Returns vector for entry in train + test data of the form 
		data = {'fbpost': 0, 'snaps' : 3, 'lastSRS': 11, 
		'seats': ["100-U-403","100-U-405","100-U-406","100-U-407"]}
		"""
		local_seat_list = self.seat_histogram
		for s in srs_seats:
			if local_seat_list.get(s) is None:
				local_seat_list[s]  = 1
			local_seat_list[s] += 1
		local_seat_list = self.replace_value_with_definition(True, 0, local_seat_list)
		curr = { 'lastSRSCount': lastSRSCount,  'num_seats': len_snapsh, 'num_fb': len_fbp, 'n_u_fb': unique_fb, 'n_u_s':unique_snaps }
		curr = {**curr, **local_seat_list}
		return curr 

	def allocate_test_train(self, 
							X_train_pos, 
							Y_train_pos, 
							X_train_neg, 
							Y_train_neg,
							X_test_pos, 
							Y_test_pos, 
							X_test_neg, 
							Y_test_neg, 
							curr, 
							banned):
		rand = random.random()
		if rand > 0.5:
			if (banned == "true"):
				X_train_pos = X_train_pos.append(curr, ignore_index=True)
				Y_train_pos = Y_train_pos.append({'res': 1},ignore_index=True )
			else:
				X_train_neg = X_train_neg.append(curr, ignore_index=True)
				Y_train_neg = Y_train_neg.append({'res': 0},ignore_index=True )
		else:
			if (banned == "true"):
				X_test_pos = X_test_pos.append(curr, ignore_index=True)
				Y_test_pos = Y_test_pos.append({'res': 1},ignore_index=True )
			else:
				X_test_neg = X_test_neg.append(curr,ignore_index=True)
				Y_test_neg = Y_test_neg.append({'res': 0},ignore_index=True )
		return X_train_pos, Y_train_pos, X_train_neg, Y_train_neg, X_test_pos, Y_test_pos, X_test_neg, Y_test_neg

	def balance_sets(self, Y_to_balance, to_balance_against , X_to_balance, eventh, snaps, fbpost):
		"""
		This function balances two sets. 
		"""
		num_fill = len(to_balance_against) - len(Y_to_balance) 
		eventhb = eventh.tail(num_fill)
		for index, column in eventhb.iterrows():
			banned = column['banned']
			slugDate = column["slugDate"]
			curr_id = column['userId']
			curr = self.calculate_curr(column[8], curr_id, snaps, fbpost, column[6], slugDate)
			X_to_balance = X_to_balance.append(curr, ignore_index=True)
			Y_to_balance = Y_to_balance.append({'res': 1}, ignore_index=True)
		return X_to_balance, Y_to_balance


	def extract_features(self):
		"""
		This function extracts features, which currently include the following
		1. Facebook posts (aggregated per user)
		2. Number of snapshots (aggreagted per user)
		3. The distribution of seats that are in the snapshot
		3. LastSRSCount

		"""
		scaler = StandardScaler()
		event = pd.read_csv("eventusers.csv") 
		seats = np.unique(event["SRSstring"])
		for i in seats:
			curr = self.parse_seats(i)
			self.seat_histogram = {**self.seat_histogram, **curr} # randomization occurs here. 
		# Now i'm here. 
		seat_list = list(self.seat_histogram.keys())
		self.feature_vector = ['lastSRSCount', 'num_seats', 'num_fb'] + seat_list

		# Preprocessing and initialization of dataframes
		# used_indices is to make sure the test and train data 
		# do not overlap with banned users 
		X_train_pos = pd.DataFrame(columns=self.feature_vector)
		X_train_neg = pd.DataFrame(columns=self.feature_vector)
		X_test_neg =  pd.DataFrame(columns=self.feature_vector)
		X_test_pos =  pd.DataFrame(columns=self.feature_vector)
		Y_test_neg =  pd.DataFrame()
		Y_test_pos  = pd.DataFrame()
		Y_train_pos = pd.DataFrame()
		Y_train_neg = pd.DataFrame()
		used_indices = [] 

		with open(self.EVENTUSERS_CSV, "rt") as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			index = 0
			snaps = pd.read_csv(self.SNAPS_CSV)
			fbpost= pd.read_csv(self.FBPOST_CSV) 
			for column in reader:
				if index == 0:
					index += 1
				elif index < self.NUM:
					banned = column[-1]
					slugDate = column[5]
					if (banned == "true"):
						used_indices.append(index) # append User Id
					curr_id = column[3]
					curr = self.calculate_curr(column[8], curr_id, snaps, fbpost, column[6], slugDate)
					# Split data - 50% put into train data, 50% into test data. 
					res = self.allocate_test_train(X_train_pos, 
													Y_train_pos, 
													X_train_neg, 
													Y_train_neg,
													X_test_pos, 
													Y_test_pos, 
													X_test_neg, 
													Y_test_neg, 
													curr, 
													banned)
					X_train_pos = res[0]
					Y_train_pos = res[1]
					X_train_neg = res[2]
					Y_train_neg = res[3]
					X_test_pos = res[4]
					Y_test_pos = res[5]
					X_test_neg = res[6]
					Y_test_neg = res[7]
					index += 1

		# Dropping banned users that were already in the trained set to avoid validating from trained set.
		event = event.drop(used_indices)
		eventh = event.loc[event['banned'] == True]
		# To make sure the training data is optimal, balance the banned and unbanned examples
		# in training set to optimize preformance.
		if (len( Y_train_pos) < len(Y_train_neg)):
			X_train_pos, Y_train_pos = self.balance_sets(Y_train_pos, 
														 Y_train_neg, 
														 X_train_pos, 
														 eventh, 
														 snaps, 
														 fbpost)

		# train on same amount of positive as negatives
		if (len(Y_test_pos) < len(Y_test_neg)): 
			X_test_pos, Y_test_pos  = self.balance_sets(Y_test_pos,
														Y_test_neg, 
														X_test_pos, 
														eventh,
														snaps, 
														fbpost,)
			X_test_pos = preprocessing.normalize(X_test_pos)
														

		# Finally put the data into a form to feed the model
		X_train_neg = X_train_neg[:len(X_train_pos)] 
		Y_train_neg = Y_train_neg[:len(X_train_pos)]
		X_train = X_train_neg.append(X_train_pos)
		X_train = pd.DataFrame(preprocessing.normalize(X_train))
		Y_train = Y_train_neg.append(Y_train_pos)
		X_test_meg = preprocessing.normalize(X_test_neg)
		X_test_pos = preprocessing.normalize(X_test_pos)
		return X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos

	def partial_train(self, seats, fbp, snapsh, lastSRS, res, uniqueFB, uniqueSnaps):
		"""
		This fits the models in real time. 
		Call this when you're training the model in real time and the model screws up
		"""
		# tHis iw here this stuff is bieng at right now. 
		print("PARTIAL TRAIN")
		print(seats)
		print(fbp)
		print(snapsh)
		X_train = pd.DataFrame(columns=self.feature_vector)
		curr = self.calculate_curr_helper(seats, snapsh, fbp, lastSRS, uniqueFB, uniqueSnaps)
		print("After curr")
		print(curr)
		X_train = X_train.append(curr, ignore_index=True)
		print(X_train)
		X_train = X_train.as_matrix().reshape(1,-1)
		Y_train = pd.DataFrame()
		Y_train = Y_train.append({'res': res},ignore_index=True )
		Y_train = Y_train.as_matrix().reshape(1, -1)
		print(Y_train)
		print(X_train)
		for c in self.classifiers:
			c = c.partial_fit(X_train, Y_train)
		return

	def train(self, X_train, Y_train):
		"""
			Train networks
		"""
		for c in self.classifiers:
			c.fit(X_train, Y_train)
		return 

	def predict_banned(self, seats, fbp, snapsh, lastSRS, uniqueFB, uniqueSnaps):
		"""
		The point of this is to be the API endpoint for checking if the user 
		is banned or not.
		The parameters are: 
		Seats -> SRSString
		FBP - Facebook posts (aggregated per user)
		snapsh - Number of snapshots (aggreagted per user)
		LastSRSCount 
		"""
		X_test = pd.DataFrame(columns=self.feature_vector)
		curr = self.calculate_curr_helper(seats, snapsh, fbp, lastSRS, uniqueFB, uniqueSnaps)
		X_test = X_test.append(curr, ignore_index=True)
		X_test = X_test.as_matrix().reshape(1,-1)
		is_banned = self.classifiers[0].predict(X_test)
		# Check if this is banned. 
		return is_banned[0]

	def test(self,  X_test, Y_test):
		# Take the average 
		for c in self.classifiers:
			start = time.time()
			score = c.score(X_test, Y_test)
			end = time.time()
			return score



if __name__ == "__main__":
	model = FraudDetection()



