from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import csv
import pandas as pd
import random 
import numpy as np
import ast
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
	scaler = StandardScaler()
	X_train_pos = pd.DataFrame(columns=[  'lastSRSCount', 'snooze', 'num_seats'])
	X_train_neg =  pd.DataFrame(columns=[  'lastSRSCount', 'snooze', 'num_seats'])
	X_test_neg =  pd.DataFrame(columns=[  'lastSRSCount', 'snooze', 'num_seats'])
	X_test_pos =   pd.DataFrame(columns=[  'lastSRSCount', 'snooze', 'num_seats'])
	Y_test_neg =  pd.DataFrame()
	Y_test_pos  = pd.DataFrame()
	Y_train_pos = pd.DataFrame()
	Y_train_neg = pd.DataFrame()

	with open("eventusers.csv", "rt") as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		index = 1
		snaps = pd.read_csv("snapshots.csv") 
		for row in reader:
			print("row")
			if index == 1:
				index += 1
			else:
				banned = row[-1]
				curr_id = row[0]
				SRS_string = ''
				ind = 8
				SRS_string = ''
				ind = 8
				while "]" not in row[ind]:
					SRS_string += row[ind] 
					ind += 1
				#print("done")
				SRS_string += row[ind]
				SRS_string = SRS_string.replace("\"\"\"", "\",")
				#print(SRS_string)
				SRS_string = ast.literal_eval(SRS_string)
				if (row[7] == "false"):
					sn = 0
				else:
					sn = 1

				snapsh = snaps.loc[snaps['userId'] == curr_id]		
				curr = { 'lastSRSCount': row[6], 'snooze': sn, 'num_seats': len(snapsh) }
				# Making sure the number of 
				if random.random() > 0.5:
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
				X_train_neg = X_train_neg[:len(X_train_pos)] # half neg, half pos examples
				Y_train_neg = Y_train_neg[:len(X_train_pos)]
				X_train = X_train_neg.append(X_train_pos)
				Y_train = Y_train_neg.append(Y_train_pos)
		return X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos



def nn_one(X_train, Y_train):
	#X = [[0., [0., 0.]], [1., [0., 0.] ]]
	#Y = [0,1]

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
		hidden_layer_sizes=(30,30,30), random_state=1)
	clf.fit(X_train, Y_train)
	return clf

def test(classifier_funcs, X_test, Y_test):
	for c in classifier_funcs:
		start = time.time()
		print(c.score(X_test, Y_test))
		end = time.time()
		print("Time to predict in seconds")
		print(end-start)



X_train, Y_train, X_test_neg, Y_test_neg, X_test_pos, Y_test_pos = extract_features() 
print(X_train)
print(len(X_train))
print(len(Y_train))
print("seperate data")
classifiers = []
clsfr1 = nn_one(X_train, Y_train)
print("training done")
classifiers.append(clsfr1)
print("false negatives, meaning it was a banned, and predicts not banned") 
test(classifiers, X_test_pos, Y_test_pos)
print("false  positives- says banned, is not") 
test(classifiers, X_test_neg, Y_test_neg)
# Maybe I sould be having my onw thing. iT'S ALREAYD mRACH. 



