import pandas as pd
import numpy as np
import matplotlib as plt

train= pd.read_csv("~/Documents/Analytics/AnalyticsVidhya/bigmart/trainUpdated.csv")
test= pd.read_csv("~/Documents/Analytics/AnalyticsVidhya/bigmart/testUpdated.csv")

train.columns
train.info()
Y= train['Items_Sold']
Y_Item_Identifier=test['Item_Identifier']
del train['Items_Sold']
del test['Items_Sold']

del train['Item_Identifier']
del test['Item_Identifier']


X_train = pd.get_dummies(train)
X_test = pd.get_dummies(test)

X_train.columns.equals(X_test.columns)

print(X_train.shape)
print(X_test.shape)

X_train.columns.difference(X_test.columns)
#X_test[X_train.columns.difference(X_test.columns)[0]] = 0
X_test = X_test[X_train.columns]



#Decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0, max_depth=40)
dtree.fit(X_train, Y)
Y_predicted = pd.Series(dtree.predict(X_test))

from numpy import multiply, product
sales =  multiply(Y_predicted , pd.Series( X_test['Item_MRP']))
submit=pd.DataFrame({'Item_Identifier':Y_Item_Identifier, 'Outlet_Identifier':test['Outlet_Identifier'],'Item_Outlet_Sales':sales.round()})


#Random Forest

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=350, max_depth=9)
clf = clf.fit(X_train, Y)
Y_predicted=pd.Series(clf.predict(X_test))
from numpy import multiply
sales =  multiply(Y_predicted , pd.Series( test['Item_MRP']))
submit=pd.DataFrame({'Item_Identifier':Y_Item_Identifier, 'Outlet_Identifier':test['Outlet_Identifier'],'Item_Outlet_Sales':sales.round()})
#Write to csv fro submission
path="~/Documents/Analytics/AnalyticsVidhya/BigMart/pythonSub10.csv"
submit.to_csv(path, index=False)


#AdaBoosting
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(n_estimators=350,)
ada= ada.fit(X_train, Y)
Y_predicted=pd.Series(ada.predict(X_test))
from numpy import multiply
sales =  multiply(Y_predicted , pd.Series( test['Item_MRP']))
submit=pd.DataFrame({'Item_Identifier':Y_Item_Identifier, 'Outlet_Identifier':test['Outlet_Identifier'],'Item_Outlet_Sales':sales.round()})
#Write to csv fro submission
path="~/Documents/Analytics/AnalyticsVidhya/BigMart/pythonSub10.csv"
submit.to_csv(path, index=False)

from sklearn.ensemble import GradientBoostingRegressor
gra = GradientBoostingRegressor(n_estimators=350,loss='lad')
gra= gra.fit(X_train, Y)
Y_predicted=pd.Series(gra.predict(X_test))
from numpy import multiply
sales =  multiply(Y_predicted , pd.Series( test['Item_MRP']))
submit=pd.DataFrame({'Item_Identifier':Y_Item_Identifier, 'Outlet_Identifier':test['Outlet_Identifier'],'Item_Outlet_Sales':sales.round()})
#Write to csv fro submission
path="~/Documents/Analytics/AnalyticsVidhya/BigMart/pythonSub10.csv"
submit.to_csv(path, index=False)

