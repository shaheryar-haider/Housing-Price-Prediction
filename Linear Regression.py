import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv ("housing.csv")

from sklearn.model_selection import train_test_split
X = data.drop (['median_house_value'], axis=1)
y = data ['median_house_value']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

train_data = X_train.join (y_train)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, y_train = train_data.drop(['median_house_value'],axis=1), train_data['median_house_value']
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()

reg.fit (X_train_s,y_train)

test_data = X_train.join(y_train)
test_data ['total_rooms'] = np.log(train_data['total_rooms']+1)
test_data ['total_bedrooms'] = np.log(train_data['total_bedrooms']+1)
test_data ['population'] = np.log(train_data['population']+1)
test_data ['households'] = np.log(train_data['households']+1)

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
test_data['bedroom_ratio'] = train_data ['total_bedrooms']/train_data['total_rooms']
test_data ['household_rooms'] = train_data ['total_rooms']/train_data['households']

X_test, y_test = test_data.drop(['median_house_value'],axis=1), test_data['median_house_value']
X_test_s = scaler.transform(X_test)
r_squared = reg.score(X_test_s, y_test)
print(f"R-squared score : {r_squared}")