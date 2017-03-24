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
""" Goal: Detect fruad, where requesting changes in stadium > taking pictures < sharing on FB"""
# Shindig at 5 pm on Saturday
# Try various appraoches to neaurl networks - feeding all 3 into neural networks,, 
# Feed into one neural network.. 

# Not using facebook or twitwter yet. 
""" Ways to approach:
1. Neural networks with the current fraud
2. One neural net for each feature, then join in a massive NN
3. Use others, like random forest
4. nueral network with the the number of seats - onehotencoder 
5. Use differences in the euclidean distance (for distance)
6. Do num for facebook posts
"""

class FraudDetection:
	NUM = 100
	def __init__(self):
		classifiers = []
		X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos = extract_features() 
		clsfr1 = nn_one(X_train, Y_train)
		classifiers.append(clsfr1)
		test(classifiers, X_test_pos, Y_test_pos)
		

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

	def extract_features(self):
		"""
		This function extracts features 
		Check which positive sare already used

		"""
		scaler = StandardScaler()
		event = pd.read_csv("eventusers.csv") 
		seats = np.unique(event["SRSstring"])
		ind_seats = {}
		for i in seats:
			curr = parse_seats(i)
			ind_seats = {**ind_seats, **curr}
		# Now i'm here. 
		seat_list = list(ind_seats.keys())
		feature_vector = ['lastSRSCount', 'num_seats', 'num_fb'] + seat_list

		# Preprocessing for onehotENcoding - where the columns represent the time stmap 
		X_train_pos = pd.DataFrame(columns=feature_vector)
		X_train_neg = pd.DataFrame(columns=feature_vector)
		X_test_neg =  pd.DataFrame(columns=feature_vector)
		X_test_pos =  pd.DataFrame(columns=feature_vector)
		Y_test_neg =  pd.DataFrame()
		Y_test_pos  = pd.DataFrame()
		Y_train_pos = pd.DataFrame()
		Y_train_neg = pd.DataFrame()
		used_indices = []
		# TRy first without ordinal data, so it's fided size 
		# THen if it doesn't work then doing  ordinal 
		# there's. alot of zeros. 
		# ther'es too many zeros. 
		with open("eventusers.csv", "rt") as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			index = 0
			snaps = pd.read_csv("snapshots.csv")
			fbpost= pd.read_csv("facebookposts.csv") 
			for row in reader:
				print(index)
				if index == 0:
					index += 1
				elif index < NUM:
					banned = row[-1]
					if (banned == "true"):
						used_indices.append(index) # append User Id
					curr_id = row[0]
					if (row[7] == "false"):
						sn = 0
					else:
						sn = 1
					# getting teh number of seats taken 
					local_seat_list = ind_seats
					local_seats = row[8] 
					seats_ordered = parse_seats(local_seats)
					#print(seats_ordered)
					for s in seats_ordered:
						if local_seat_list.get(s) is None:
							local_seat_list[s]  = 1
						local_seat_list[s] += 1
					local_seat_list = replace_value_with_definition(True, 0, local_seat_list)
					#print(local_seat_list)
					# here, you go there as well. 
					snapsh = snaps.loc[snaps['userId'] == curr_id]	
					fbp = fbpost.loc[snaps['userId'] == curr_id]	

					curr = { 'lastSRSCount': row[6],  'num_seats': len(snapsh), 'num_fb': len(fbp)  }
					curr = {**curr, **local_seat_list}
					# Making sure the number of 
					# MAKE SURE THERE IS A ONE HERE. 
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
					index += 1

		# Dropping this. 
		event = event.drop(used_indices)
		eventh = event.loc[event['banned'] == True]
		# NOW, what you want to do is make it an ordnal data. 
		if (len( Y_train_pos) < len(Y_train_neg)):
			num_fill = len(Y_train_neg) - len( Y_train_pos) 
			eventha = eventh.head(num_fill)
			for index, row in eventha.iterrows():
				banned = row['banned']
				curr_id = row['_id']
				if (row[7] == 'False'):
					sn = 0
				else:
					sn = 1
				snapsh = snaps.loc[snaps['userId'] == curr_id]	
				fbp = fbpost.loc[snaps['userId'] == curr_id]
				# can this get [] data. 
				local_seats = row[8] 
				local_seat_list = ind_seats
				local_seats = row[8] 
				seats_ordered = parse_seats(local_seats)
				#print(seats_ordered)
				for s in seats_ordered:
					if local_seat_list.get(s) is None:
						local_seat_list[s]  = 1
					local_seat_list[s] += 1
				local_seat_list = replace_value_with_definition(True, 0, local_seat_list)
				curr = { 'lastSRSCount': row[6],  'num_seats': len(snapsh), 'num_fb': len(fbp)  }
				curr = {**curr, **local_seat_list}
				X_train_pos = X_train_pos.append(curr, ignore_index=True)
				Y_train_pos = Y_train_pos.append({'res': 1},ignore_index=True )
		if (len(Y_test_pos) < len(Y_test_neg)):
			num_fill = len(Y_test_neg) - len( Y_test_pos) 
			eventhb = eventh.tail(num_fill)
			for index, row in eventhb.iterrows():
				banned = row['banned']
				curr_id = row['_id']
				if (row[7] == 'False'):
					sn = 0
				else:
					sn = 1
				snapsh = snaps.loc[snaps['userId'] == curr_id]	
				fbp = fbpost.loc[snaps['userId'] == curr_id]
				local_seats = row[8] 
				local_seat_list = ind_seats
				local_seats = row[8] 
				seats_ordered = parse_seats(local_seats)
				#print(seats_ordered)
				for s in seats_ordered:
					if local_seat_list.get(s) is None:
						local_seat_list[s]  = 1
					local_seat_list[s] += 1
				local_seat_list = replace_value_with_definition(True, 0, local_seat_list)
				curr = { 'lastSRSCount': row[6],  'num_seats': len(snapsh), 'num_fb': len(fbp)  }
				curr = {**curr, **local_seat_list}
				X_test_pos = X_test_pos.append(curr, ignore_index=True)
				Y_test_pos = Y_test_pos.append({'res': 1},ignore_index=True )

		X_train_neg = X_train_neg[:len(X_train_pos)] # half neg, half pos examples
		Y_train_neg = Y_train_neg[:len(X_train_pos)]
		X_train = X_train_neg.append(X_train_pos)
		Y_train = Y_train_neg.append(Y_train_pos)
		return X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos


	def train(self, X_train, Y_train):
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
			hidden_layer_sizes=(30,30,30), random_state=1)
		clf.fit(X_train, Y_train)
		return clf

	def test(self, classifier_funcs, X_test, Y_test):
		for c in classifier_funcs:
			start = time.time()
			print(c.score(X_test, Y_test))
			end = time.time()
			print("Time to predict in seconds")
			print(end-start)

	# Sure I'll be there soon. 
# Here, it'll go from 4:45 - 5 pm


