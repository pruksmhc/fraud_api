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
	NUM = 100
	EVENTUSERS_CSV = "eventusers.csv"
	SNAPS_CSV = "snapshots.csv"
	FBPOST_CSV = "facebookposts.csv"

	def __init__(self):
		self.classifiers = []
		X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos = self.extract_features() 
		classifier1 = MLPClassifier(solver='lbfgs', alpha=1e-5,
			hidden_layer_sizes=(30,30,30), random_state=1)
		self.classifiers.append(classifier1)
		self.train(X_train, Y_train)
		self.test(X_test_pos, Y_test_pos)

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

	def calculate_curr(self, row, curr_id, snaps, fbpost, ind_seats):
		"""
		This function, from the CSVs, extracts the necessary data and puts it 
		into curr, which is the form of the training data to be fed in. 
		"""
		local_seat_list = ind_seats
		local_seats = row[8] 
		seats_ordered = self.parse_seats(local_seats)
		for s in seats_ordered:
			if local_seat_list.get(s) is None:
				local_seat_list[s]  = 1
			local_seat_list[s] += 1
		local_seat_list = self.replace_value_with_definition(True, 0, local_seat_list)
		snapsh = snaps.loc[snaps['userId'] == curr_id]	
		fbp = fbpost.loc[fbpost['userId'] == curr_id]	
		curr = { 'lastSRSCount': row[6],  'num_seats': len(snapsh), 'num_fb': len(fbp)}
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

	def balance_sets(self, Y_to_balance, to_balance_against , X_to_balance, eventh, snaps, fbpost, ind_seats):
		"""
		This function balances two sets. 
		"""
		num_fill = len(to_balance_against) - len(Y_to_balance) 
		eventhb = eventh.tail(num_fill)
		for index, row in eventhb.iterrows():
			banned = row['banned']
			curr_id = row['_id']
			curr = self.calculate_curr(row, curr_id, snaps, fbpost, ind_seats)
			X_test_pos = X_to_balance.append(curr, ignore_index=True)
			Y_test_pos = Y_to_balance.append({'res': 1}, ignore_index=True)
		return X_test_pos, Y_test_pos


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
		ind_seats = {}
		for i in seats:
			curr = self.parse_seats(i)
			ind_seats = {**ind_seats, **curr}
		# Now i'm here. 
		seat_list = list(ind_seats.keys())
		feature_vector = ['lastSRSCount', 'num_seats', 'num_fb'] + seat_list

		# Preprocessing and initialization of dataframes
		# used_indices is to make sure the test and train data 
		# do not overlap with banned users 
		X_train_pos = pd.DataFrame(columns=feature_vector)
		X_train_neg = pd.DataFrame(columns=feature_vector)
		X_test_neg =  pd.DataFrame(columns=feature_vector)
		X_test_pos =  pd.DataFrame(columns=feature_vector)
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
					curr = self.calculate_curr(row, curr_id, snaps, fbpost, ind_seats)
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
														 fbpost, 
														 ind_seats)
		if (len(Y_test_pos) < len(Y_test_neg)):
			X_test_pos, Y_test_pos  = self.balance_sets(Y_test_pos,
														Y_test_neg, 
														X_test_pos, 
														eventh,
														snaps, 
														fbpost, 
														ind_seats)

		# Finally put the data into a form to feed the model
		X_train_neg = X_train_neg[:len(X_train_pos)] 
		Y_train_neg = Y_train_neg[:len(X_train_pos)]
		X_train = X_train_neg.append(X_train_pos)
		Y_train = Y_train_neg.append(Y_train_pos)
		return X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos

	def partial_train(self, X, Y):
		"""
		This fits the models in real time
		"""
		for c in self.classifiers:
			c.partial_fit(X, Y)

	def train(self, X_train, Y_train):
		"""
			Train networks
		"""
		for c in self.classifiers:
			c.fit(X_train, Y_train)
		return 

	def predict_banned(self, user_id, fbpost, snaps, seats, lastSRS):
		"""
		The point of this is to be the API endpoint for checking if the user 
		is banned or not.
		The parameters are: 
		1. Facebook posts (aggregated per user)
		2. Number of snapshots (aggreagted per user)
		3. The distribution of seats that are in the snapshot
		3. LastSRSCount
		"""
		curr = self.calculate_curr(row, curr_id, snaps, fbpost, ind_seats)
		model_input.append(curr)  
		is_banned = self.classifiers[0].predict(model_input)
		# Check if this is banned. 
		return is_banned 

	def test(self,  X_test, Y_test):
		# Take the average 
		for c in self.classifiers:
			start = time.time()
			print("accuracy score")
			print(c.score(X_test, Y_test))
			end = time.time()
			print("Time to predict in seconds")
			print(end-start)

	# now you do the API and then call itfrom ehre. 
if __name__ == "__main__":
	model = FraudDetection()


