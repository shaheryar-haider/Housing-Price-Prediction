
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv ("housing.csv")

data.dropna(inplace=True)

data.info()

from sklearn.model_selection import train_test_split
X = data.drop (['median_house_value'], axis=1)
y = data ['median_house_value']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

train_data = X_train.join (y_train)
#train_data.hist(figsize=(15,8))



train_data ['total_rooms'] = np.log(train_data['total_rooms']+1)
train_data ['total_bedrooms'] = np.log(train_data['total_bedrooms']+1)
train_data ['population'] = np.log(train_data['population']+1)
train_data ['households'] = np.log(train_data['households']+1)



train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
train_data
#plt.figure (figsize =(15,8))
#sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")

#plt.figure(figsize=(15,8))
#sns.scatterplot(x='latitude',y='longitude',data=train_data, hue="median_house_value", palette= "coolwarm")

train_data['bedroom_ratio'] = train_data ['total_bedrooms']/train_data['total_rooms']
train_data ['household_rooms'] = train_data ['total_rooms']/train_data['households']

plt.figure (figsize =(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")

from sklearn.linear_model import LinearRegression

X_train, y_train = train_data.drop(['median_house_value'],axis=1), train_data['median_house_value']
reg = LinearRegression()
reg.fit (X_train,y_train)
test_data = X_train.join (y_train)
test_data ['total_rooms'] = np.log(train_data['total_rooms']+1)
test_data ['total_bedrooms'] = np.log(train_data['total_bedrooms']+1)
test_data ['population'] = np.log(train_data['population']+1)
test_data ['households'] = np.log(train_data['households']+1)

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
test_data['bedroom_ratio'] = train_data ['total_bedrooms']/train_data['total_rooms']
test_data ['household_rooms'] = train_data ['total_rooms']/train_data['households']


