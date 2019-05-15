#LOADING YOUR DATASET
import os
import pandas as pd

dataset_dir = "your_path_to_dir"

def dataset_load (my_dir=dataset_dir):
    csv_file = os.path.join(my_dir, "USD_EUR Historical Data.csv")
    return pd.read_csv(csv_file)
ue_data = data_load()

#SEE THE FIRST NINE ROWS FROM THE DATASET
ue_data.head()
"""This gives us the following columns: Date, Open, High, Low, Price, Change_percentage"""

#CLEANING THE DATA
import datetime as dt
cdata = ue_data[["Date", "Open", "High", "Low", "Price", "Change %"]]
cdata["Date"] = pd.to_datetime(cdata["Date"])
cdata["Date"] = cdata["Date"].map(dt.datetime.toordinal)
cdata["Change_percentage"] = cdata["Change %"]
i=0
for cd in cdata["Change_percentage"]:
    cdata["Change_percentage"][i] = cd.replace("%", "")
    i=i+1
cdata = cdata[["Date", "Open", "High", "Low", "Price", "Change_percentage"]]
cdata.head()

#PLOTTING TARGET WITH AN INDEPENDANT VARIABLE
import matplotlib.pyplot as plt
plt.scatter(cdata.Date, cdata.Price, color="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()



#CREATING TRAIN AND TEST SPLIT
import numpy as np
msk = np.random.rand(len(cdata)) < 0.8
train_set = cdata[msk]
test_set = cdata[~msk]


#TRAIN DATA DISTRIBUTION
plt.scatter(train_set.Date, train_set.Price, color="yellow")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


#MULTIPLE REGRESSION MODULE USING THE TEST SET
from sklearn import linear_mode
reg = linear_model.LinearRegression()
xtest = np.asanyarray(test_set[["Date", "Open", "High", "Low"]])
ytest = np.asanyarray(test_set[["Price"]])
reg.fit(xtest, ytest)
print("Coefficient = ", reg.coef_)

#PREDICTION
from sklearn.metrics import r2_score
ytest_predict = reg.predict(test_set[["Date", "Open", "High", "Low"]])
print("MSE -> ", np.mean(ytest_predict - ytest)**2))
print("R2_SCORE -> ", r2_score(ytest_predict, ytest))
i=0
for el in ytest_predict:
    print (ytest[i], "->", el)
    i=i+1

