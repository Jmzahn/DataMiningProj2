import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats import weightstats as stests
import statsmodels.tools.tools as stattools
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
np.random.seed(12345678)

print('Loading churn...')
churn = pd.read_csv('./churn')
print('Done!')

#1. Using the churn data set, determine how many records need to be resampled in order to
#   have 20% of the rebalanced data set have true Churn variable values. Create the rebalanced
#   data set and confirm that the new data set has 20% true Churn variable values.

#first lets print the churn['Churn'] values

print(churn['Churn'].value_counts())

#now lets grab the numbers
numFalse, numTrue = churn['Churn'].value_counts()
numToResample = (0.20*(numFalse+numTrue)-numTrue)/0.80
print('Number of values needed to be resampled: ',numToResample)
print('This would result in ',(numToResample+numTrue),' True values, which would be ',(100*((numToResample+numTrue)/(numToResample+numTrue+numFalse))),'%')
print('Obviusly we cant resample a half of a sample so we either round up or down.')


#The book warns us to never balance test data so before we go about balancing im going to skip 
# to question 2 and split the data, then balance the training data

#2. Partition the rebalanced data set so that 67% of the records are included in the training
#   data set and 33% are included in the test data set. Use a bar graph to confirm the 
#   proportions. Validate the training and test data sets by testing for the difference in the
#   training and test means using Day.Mins (t-test) and the Z-test on the Churn variable. Try
#   forming new training and test sets if there is enough evidence to reject the null hypothesis.

churnTrain, churnTest = train_test_split(churn, test_size = 0.33, random_state = 2)
print('churn Shape: ',churn.shape)
print('churnTrain Shape: ',churnTrain.shape)
print('churnTest Shape: ',churnTest.shape)
y=churn.shape[0]
plt.barh(y,churnTrain.shape[0],color='b', alpha=0.60)
plt.barh(y,churnTest.shape[0],left=churnTrain.shape[0],color='r', alpha=0.60)
plt.title('Bar graph of churn Train/Test split')
plt.yticks([])
plt.xlabel('Number of samples')
plt.show()

ttest, pval1 = ttest_ind(churnTrain['Day Mins'], churnTest['Day Mins'])
if pval1<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
ztest, pval2 = stests.ztest(churnTrain['Churn'], x2=churnTest['Churn'], value=0,alternative='two-sided')
if pval2<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

#Our t-test indicated we should reject the null hypothesis so lets resplit
churnTrain, churnTest = train_test_split(churn, test_size = 0.33)
ttest, pval1 = ttest_ind(churnTrain['Day Mins'], churnTest['Day Mins'])
if pval1<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
ztest, pval2 = stests.ztest(churnTrain['Churn'], x2=churnTest['Churn'], value=0,alternative='two-sided')
if pval2<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
#No indication that we should reject the null hypothesis

#Now that we have our train/test split we can rebalance our training data 
#get True/False counts
print(churnTrain['Churn'].value_counts())
numFalse, numTrue = churnTrain['Churn'].value_counts()
#calculate x
numToResample = (0.20*(numFalse+numTrue)-numTrue)/0.80
print('Number of values needed to be resampled: ',numToResample)
print('This would result in ',(numToResample+numTrue),' True values, which would be ',(100*((numToResample+numTrue)/(numToResample+numTrue+numFalse))),'%')
print('Obviusly we cant resample a half of a sample so we either round up or down.')
#create resample pool and samples then concatenate samples and original
toResample = churnTrain.loc[churnTrain['Churn']==True]
ourResample = toResample.sample(n=int(numToResample), replace = True)
churnTrainRebal = pd.concat([churnTrain, ourResample])

#3. Create  a  CART  model  using  the  training  set  with  the Churntarget  variable  and
#   whatever  predictor  variables you  think  appropriate.  Try at least 3 different models.
#   Compare the confusion tables and accuracies of the 3 different models.

y = churnTrainRebal['Churn'].to_numpy()
yTest = churnTest['Churn'].to_numpy()
#drop the Churn col from the dataset to obtain X

X = churnTrainRebal.drop('Churn',axis=1)
XTest = churnTest.drop('Churn',axis=1)
#drop all categorical cols so we can replace them with dummy vals if we want
toDrop = ['State', 'Phone', 'Intl Plan', 'VMail Plan', 'Old Churn']
#I plan to readd 'Intl Plan', 'VMail Plan', 'Old Churn' as dummy vals
X.drop(toDrop, axis=1, inplace=True)
XTest.drop(toDrop, axis=1, inplace=True)
xNames = X.columns.values
yNames = ['True','False']
cart01 = DecisionTreeClassifier(criterion="gini",max_leaf_nodes=2).fit(X,y)
export_graphviz(cart01, out_file='cart01.dot', feature_names=xNames, class_names=yNames)
cart02 = DecisionTreeClassifier(criterion="gini",max_leaf_nodes=5).fit(X,y)
export_graphviz(cart02, out_file='cart02.dot', feature_names=xNames, class_names=yNames)
cart03 = DecisionTreeClassifier(criterion="gini",max_leaf_nodes=10).fit(X,y)
export_graphviz(cart02, out_file='cart03.dot', feature_names=xNames, class_names=yNames)
#only difference is max_leaf_nodes
#lets collect our predictions
predCart01 = cart01.predict(XTest)
predCart02 = cart02.predict(XTest)
predCart03 = cart03.predict(XTest)
cm01 = confusion_matrix(yTest,predCart01)
cm02 = confusion_matrix(yTest,predCart02)
cm03 = confusion_matrix(yTest,predCart03)
print('\nConfusion matrices and accuracies')
print('cm01:\n',cm01)
print('Accuracy: ',accuracy_score(yTest,predCart01))
print('cm02:\n',cm02)
print('Accuracy: ',accuracy_score(yTest,predCart02))
print('cm03:\n',cm03)
print('Accuracy: ',accuracy_score(yTest,predCart03))

#4. Create a C5.0 model using the training set with the Churntarget variable and
#   whatever predictor variables you think  appropriate. Try at least 3 different models.
#   Compare the confusion tables and accuracies of the 3 different models.
#   How does C5.0 compare in performance to CART? 

c50_01 = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=2).fit(X,y)
export_graphviz(cart01, out_file='c50_01.dot', feature_names=xNames, class_names=yNames)
c50_02 = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=5).fit(X,y)
export_graphviz(cart02, out_file='c50_02.dot', feature_names=xNames, class_names=yNames)
c50_03 = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=10).fit(X,y)
export_graphviz(cart02, out_file='c50_03.dot', feature_names=xNames, class_names=yNames)
#only difference is max_leaf_nodes
#lets collect our predictions
predc50_01 = c50_01.predict(XTest)
predc50_02 = c50_02.predict(XTest)
predc50_03 = c50_03.predict(XTest)
cm01 = confusion_matrix(yTest,predc50_01)
cm02 = confusion_matrix(yTest,predc50_02)
cm03 = confusion_matrix(yTest,predc50_03)
print('\nConfusion matrices and accuracies')
print('cm01:\n',cm01)
print('Accuracy: ',accuracy_score(yTest,predc50_01))
print('cm02:\n',cm02)
print('Accuracy: ',accuracy_score(yTest,predc50_02))
print('cm03:\n',cm03)
print('Accuracy: ',accuracy_score(yTest,predc50_03))