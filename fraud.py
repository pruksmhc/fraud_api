from sklearn.neural_network import MLPClassifier
import csv
import pandas as pd
import random 
import numpy as np
import time
""" Goal: Detect fruad, where requesting changes in stadium > taking pictures < sharing on FB"""
# Shindig at 5 pm on Saturday
# Try various appraoches to neaurl networks - feeding all 3 into neural networks,, 
# Feed into one neural network.. 

# Not using facebook or twitwter yet. 
""" Ways to approach:
1. Neural networks with teh current fraud
2. One neural net for each feature, then join in a massive NN
3. Use others, like random forest

"""
def extract_features():
	X_train= []
	Y_train = []
	X_test_neg = []
	X_test_pos = []
	Y_test_pos = []
	Y_test_neg = []
	with open("eventusers.csv", "rb") as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		Fraud = [] # Here, tuple (id, users, number, et.c. )
		next(reader, None)
		snaps = pd.read_csv("snapshots.csv") 
		for row in reader:
			print("Row")
			# 3, 6, 8, 10 for banned
			curr_id = row[3] 
			SRS_seq = row[6]
			i = 10
			banned = row[-1]
			#print(SRS_seq)
			#print(SRS_num)
			snaps = snaps.loc[snaps['userId'] == curr_id]
			snap_id = []
			seats = {}
			for index, row in snaps.iterrows():
				seat = row.iloc[0]
				seat = seat.split("-")
				seat_num = ""
				to_compare = ""
				for j in range(len(seat)):
					if "cam" in seat[j]:
						to_compare = seat[j+2] + seat[j+3]
						seat_num = seat[j+1] + seat[j+2] + seat[j+3]
					if (seats.get(seat_num)):
						seats[seat_num] += 1
					else:
						seats[seat_num] = 1 
				to_compare_with = row["row"] + str(row["seat"])
				vector_final =  [len(seats)]

				if (to_compare == to_compare_with):
					vector_final.append(1) # True - same seat that tehy're in
				else:
					vector_final.append(-1) 

			### HERE, THIS IS WHERE WE PUT WEHAT WE HAVE EXTRACTED 
			### INTO A STRING
			
			for k in seats.values():
				vector_final.append(k)
			# I'm here rn 
			if random.random() > 0.5:
				X_train.append(vector_final)

				if (banned == "true"):
					Y_train.append(1)
				else:
					Y_train.append(0)
			else:
				if (banned == "true"):
					X_test_pos.append(vector_final)
					Y_test_pos.append(1)
				else:
					X_test_neg.append(vector_final)
					Y_test_neg.append(0)
		return X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos



def nn_one(X_train, Y_train):
	#X = [[0., [0., 0.]], [1., [0., 0.] ]]
	#Y = [0,1]
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
		hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train, Y_train)
	return clf


def test(classifier_funcs, X_test, Y_test):
	for c in classifier_funcs:
		start = time.time()
		predictions = np.array(c.predict(X_test))
        end = time.time()
        print("Time to predict in seconds")
        print(end-start)
        error = np.mean( predictions != Y_test)
        print("Percentage wrong")
        print(error)
        print(predictions)


X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos = extract_features()
#X_train = np.array(X_train).reshape((len(X_train), 1))
#Y_train = np.array(Y_train).reshape((len(Y_train), 1))
print(len(X_train))
print(len(Y_train))
print("seperate data")
classifiers = []
clsfr1 = nn_one(X_train, Y_train)
print("training done")
classifiers.append(clsfr1)
print("now testing")
print("false negatives, meaning it was a banned, and predicts not banned") 
test(classifiers, X_test_pos, Y_test_pos)
print("false  positives- says banned, is not") 
test(classifiers, X_test_neg, Y_test_neg)



