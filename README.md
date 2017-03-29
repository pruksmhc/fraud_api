
Documentation:
Right now, there are 2 functions. 
1. /fraud - pass in the features that are listed in the documentation in fraud_API_endpoint.py and 
	get a 1.0 or 0.0 for banned or not 
2. /partial_fit - if the model is wrong, you can train it in real time by feeding it the features 
and the human-made classification

In the future, this model will have to be trained for a few more sports games, with the flow being below. When a 
suspicious box pops up, update the MongoDB, parse the data into the inputs to the model, and feed it through. 
Then partially fit it to train the model in real time.

![Alt text](URM.jpg?raw=true "URM of prediction flow")
