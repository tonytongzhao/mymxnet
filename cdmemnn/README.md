#Collaborative Dynamic MeMNN
Parameter tuning and testing...\n
Val RMSE: 0.8401\n
python cdmemnn.py -model colgru -train ~/Data/ml-1m/ratings.dat.random.tr -val ~/Data/ml-1m/ratings.dat.random.te -nhidden 550 -nembed 550 -upass 1 -ipass 1 -eta 0.03 -nepoch 200
