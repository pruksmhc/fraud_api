from sklearn.neural_network import MLPClassifier
import csv
import pandas as pd
import random 
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
	X_test = []
	Y_test = []
	with open("eventusers.csv", "rb") as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		Fraud = [] # Here, tuple (id, users, number, et.c. )
		next(reader, None)
		for row in reader:
			# 3, 6, 8, 10 for banned
			curr_id = row[3] 
			SRS_seq = row[6]
			SRS_num = row[8]
			SRS_num += row[9]
			i = 10
			while "]" not in row[i]:
				SRS_num += row[i]
				i += 1
			SRS_num += row[i]
			banned = row[-1]
			#print(SRS_seq)
			#print(SRS_num)
			curr_info = [SRS_seq, SRS_num]
			snaps = pd.read_csv("snapshots.csv")
			snaps = snaps.loc[snaps['userId'] == curr_id]
			snap_id = []
			for index, row in snaps.iterrows():
				seat = row.iloc[0]
				seat = seat.split("-")
				seat_num = ""
				to_compare = ""
				for j in range(len(seat)):
					if "cam" in seat[j]:
						to_compare = seat[j+2] + seat[j+3]
						seat_num = seat[j+1] + seat[j+2] + seat[j+3]
						 
				to_compare_with = row["row"] + str(row["seat"])
				if (to_compare == to_compare_with):
					snap_id.append((seat_num + "True")) # True - same seat that tehy're in
				else:
					snap_id.append((seat_num + "False"))

			### HERE, THIS IS WHERE WE PUT WEHAT WE HAVE EXTRACTED 
			### INTO A STRING
			vector_string = "".join(SRS_seq)+ str(SRS_num) + "".join(snap_id)
			if random.random() > 0.5:
				X_train.append(vector_string)
				if (banned == "True"):
					Y_train.append(1);
				else:
					Y_train.append(0);
			else:
				X_test.append(vector_string)
				if (banned == "True"):
					Y_test.append(1);
				else:
					Y_test.append(0);
		return X_train, Y_train, X_test, Y_test

			


def nn_one(X_train, Y_train):
	X = [[0., [0., 0.]], [1., [0., 0.] ]]
	Y = [0,1]
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
		hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X, Y)
	return clf


def test(classifier_funcs, X_test, Y_test):
	for c in range(classifier_funcs):
		start = time.time()
		predictions = np.array(c.predict(X_test))
        end = time.time()
        print("Time to predict in seconds")
        print(end-start)
        error = np.mean( predictions != Y_test)
        print("Percentage wrong")
        print(error)
        print(predictions)



X_train, Y_train, X_test, Y_test = extract_features()
classifiers = []
clsfr1 = nn_one(X_train, Y_train)
classifiers.append(clsfr1)
test(classifiers, X_test, Y_test)


