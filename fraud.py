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
		self.classifiers = []
		self.ind_seats = {}
		self.feature_vector = []
		X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos = self.extract_features() 
		classifier1 = MLPClassifier(solver='adam', alpha=1e-5,
			hidden_layer_sizes=(30,30,30), random_state=1)
		self.classifiers.append(classifier1)
		self.train(X_train, Y_train)
		print("the accuracy for predicting banned")
		self.test(X_test_pos, Y_test_pos)
		print("the accuarcy for predicting not banned")
		self.test(X_test_neg, Y_test_neg)

	def parse_seats(self, seats):
		indiv_seats = {}
		seat_num = seats.split(",")
		for s in seat_num:
			s = s.replace("[", "")
			s = s.replace("]", "")
			s = s.split("\"")
			for se in s:
				if ((se is not "\"") and (se is not " ") and (se is not "")):
					if indiv_seats.get(se) is None:
						indiv_seats[se] = True
		return indiv_seats

	def replace_value_with_definition(self, value_to_find, definition, current_dict):
	    for key, value in current_dict.items():
	        if value == value_to_find:
	            current_dict[key] = definition
	    return current_dict

	def calculate_curr(self, seats, curr_id, snaps, fbpost, lastSRSCount):
		"""
		This function, from the CSVs, extracts the necessary data and puts it 
		into curr, which is the form of the training data to be fed in. 
		"""
		snapsh = snaps.loc[snaps['userId'] == curr_id]	
		fbp = fbpost.loc[fbpost['userId'] == curr_id]
		seats = self.parse_seats(seats)
		return self.calculate_final(seats, len(snapsh), len(fbp), lastSRSCount)

	def calculate_final(self, seats_ordered, len_snapsh, len_fbp, lastSRSCount):
		"""Ind_seats is a map : seat -> number
			Seats is the string of seat SRS"""
		local_seat_list = self.ind_seats
		for s in seats_ordered:
			if local_seat_list.get(s) is None:
				local_seat_list[s]  = 1
			local_seat_list[s] += 1
		local_seat_list = self.replace_value_with_definition(True, 0, local_seat_list)
		curr = { 'lastSRSCount': lastSRSCount,  'num_seats': len_snapsh, 'num_fb': len_fbp }
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
		print("num to fill")
		print(num_fill)
		eventhb = eventh.tail(num_fill)
		for index, row in eventhb.iterrows():
			banned = row['banned']
			curr_id = row['_id']
			curr = self.calculate_curr(row[8], curr_id, snaps, fbpost, row[6])
			X_to_balance = X_to_balance.append(curr, ignore_index=True)
			Y_to_balance = Y_to_balance.append({'res': 1}, ignore_index=True)
		print("and what we have")
		print(X_to_balance)
		print(Y_to_balance)
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
			self.ind_seats = {**self.ind_seats, **curr}
		# Now i'm here. 
		seat_list = list(self.ind_seats.keys())
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
			for row in reader:
				if index == 0:
					index += 1
				elif index < self.NUM:
					banned = row[-1]
					if (banned == "true"):
						used_indices.append(index) # append User Id
					curr_id = row[0]
					curr = self.calculate_curr(row[8], curr_id, snaps, fbpost, row[6])
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
		if (len(Y_test_pos) < len(Y_test_neg)):
			X_test_pos, Y_test_pos  = self.balance_sets(Y_test_pos,
														Y_test_neg, 
														X_test_pos, 
														eventh,
														snaps, 
														fbpost)
														

		# Finally put the data into a form to feed the model
		X_train_neg = X_train_neg[:len(X_train_pos)] 
		Y_train_neg = Y_train_neg[:len(X_train_pos)]
		X_train = X_train_neg.append(X_train_pos)
		Y_train = Y_train_neg.append(Y_train_pos)
		return X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos

	def partial_train(self, seats, fbp, snapsh, lastSRS, res):
		"""
		This fits the models in real time. 
		Call this when you're training the model in real time and the model screws up
		"""
		print("train")
		X_train= pd.DataFrame(columns=self.feature_vector)
		print(seats)
		curr = self.calculate_final(seats, snapsh, fbp, lastSRS)
		print("2")
		X_train= X_train.append(curr, ignore_index=True)
		Y_train = pd.DataFrame()
		Y_train = Y_train.append({'res': res},ignore_index=True )
		print("3")
		for c in self.classifiers:
			print("train")
			c = c.partial_fit(X_train, Y_train)
		return

	def train(self, X_train, Y_train):
		"""
			Train networks
		"""
		for c in self.classifiers:
			c.fit(X_train, Y_train)
		return 

	def predict_banned(self, seats, fbp, snapsh, lastSRS):
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
		curr = self.calculate_final(seats, snapsh, fbp, lastSRS)
		X_test = X_test.append(curr, ignore_index=True)
		is_banned = self.classifiers[0].predict(X_test)
		# Check if this is banned. 
		return is_banned[0]

	def test(self,  X_test, Y_test):
		# Take the average 
		for c in self.classifiers:
			start = time.time()
			print("accuracy score")
			print(c.score(X_test, Y_test))
			end = time.time()
			print("Time to predict in seconds")
			print(end-start)


if __name__ == "__main__":
	model = FraudDetection()


