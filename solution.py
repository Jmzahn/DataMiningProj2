#Jacob Zahn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats import weightstats as stests
import statsmodels.tools.tools as stattools
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
import time
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

#passing test_size = 0.33 to train_test_split should result in 67/33 train/test split, random_state is passed for reproducibility
churnTrain, churnTest = train_test_split(churn, test_size = 0.33, random_state = 2)
#print the shapes
print('churn Shape: ',churn.shape)
print('churnTrain Shape: ',churnTrain.shape)
print('churnTest Shape: ',churnTest.shape)

#create bar graph from the shapes
y=churn.shape[0]#total
plt.barh(y,churnTrain.shape[0],color='b', alpha=0.60)#train size
plt.barh(y,churnTest.shape[0],left=churnTrain.shape[0],color='r', alpha=0.60)#test size, starting at end of train size
plt.title('Bar graph of churn Train/Test split')
plt.yticks([])
plt.xlabel('Number of samples')
plt.show()

#perform t-test
ttest, pval1 = ttest_ind(churnTrain['Day Mins'], churnTest['Day Mins'])
if pval1<0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")

#perform z-test
ztest, pval2 = stests.ztest(churnTrain['Churn'], x2=churnTest['Churn'], value=0,alternative='two-sided')
if pval2<0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")

#Our t-test indicated we should reject the null hypothesis so lets resplit
churnTrain, churnTest = train_test_split(churn, test_size = 0.33)
#perform t-test
ttest, pval1 = ttest_ind(churnTrain['Day Mins'], churnTest['Day Mins'])
if pval1<0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")

#perform z-test
ztest, pval2 = stests.ztest(churnTrain['Churn'], x2=churnTest['Churn'], value=0,alternative='two-sided')
if pval2<0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")
#No indication that we should reject the null hypothesis, I ran this a couple of times and believe that np.random.seed(12345678) at the top is what
# the most recent call of train_test_split is using, according to the docs it uses np.random when nothing is passed
#Should also not trigger the if statements on further re-runs if I have seeded the function correctly.

#Now that we have our train/test split we can rebalance our training data like was asked in question 1.

#get True/False counts
print(churnTrain['Churn'].value_counts())
numFalse, numTrue = churnTrain['Churn'].value_counts()
#calculate x
numToResample = (0.20*(numFalse+numTrue)-numTrue)/0.80
print('Number of values needed to be resampled: ',numToResample)
print('This would result in ',(numToResample+numTrue),' True values, which would be ',(100*((numToResample+numTrue)/(numToResample+numTrue+numFalse))),'%')
print('Obviusly we cant resample a half of a sample so we either round up or down.')

#create resample pool and samples then concatenate samples and original as shown in book
toResample = churnTrain.loc[churnTrain['Churn']==True]
ourResample = toResample.sample(n=int(numToResample), replace = True)#int() rounds down effectively
churnTrainRebal = pd.concat([churnTrain, ourResample])

#3. Create  a  CART  model  using  the  training  set  with  the Churntarget  variable  and
#   whatever  predictor  variables you  think  appropriate.  Try at least 3 different models.
#   Compare the confusion tables and accuracies of the 3 different models.

#collect the ys from the train and test sets
y = churnTrainRebal['Churn'].to_numpy()
yTest = churnTest['Churn'].to_numpy()

#drop the Churn col from the dataset to obtain Xs from the train and test sets
X = churnTrainRebal.drop('Churn',axis=1)
XTest = churnTest.drop('Churn',axis=1)

#drop all categorical cols so we can replace them with dummy vals if we want
toDrop = ['State', 'Phone', 'Intl Plan', 'VMail Plan', 'Old Churn']
#I plan to re-add 'Intl Plan', 'VMail Plan', 'Old Churn' as dummy vals

#this is how its shown in the book, but they only did it once. I'm about to do it 6 times.

#create numpy lists from the training data
intlPlan = X['Intl Plan'].to_numpy()
vMailPlan = X['VMail Plan'].to_numpy()
oldChurn = X['Old Churn'].to_numpy()

#create dummy vals from the categorical training data
(intlPlanCat, intlPlanCatDict) = stattools.categorical(intlPlan, drop=True, dictnames=True)
(vMailPlanCat, vMailPlanCatDict) = stattools.categorical(vMailPlan, drop=True, dictnames=True)
(oldChurnCat, oldChurnCatDict) = stattools.categorical(oldChurn, drop=True, dictnames=True)

#turn these lists into pd.DataFrames, I pass col names additionally unlike the book, this is so i can grab them
intlPlanCatPD = pd.DataFrame(intlPlanCat,columns=['intlPlanNo','intlPlanYes'])
vMailPlanCatPD = pd.DataFrame(vMailPlanCat,columns=['vMailPlanNo','vMailPlanYes'])
oldChurnCatPD = pd.DataFrame(oldChurnCat,columns=['oldChurnFalse','oldChurnTrue'])

#create numpy lists from the test data
intlPlanTest = XTest['Intl Plan'].to_numpy()
vMailPlanTest = XTest['VMail Plan'].to_numpy()
oldChurnTest = XTest['Old Churn'].to_numpy()

#create dummy vals from categorical test data
(intlPlanTestCat, intlPlanTestCatDict) = stattools.categorical(intlPlanTest, drop=True, dictnames=True)
(vMailPlanTestCat, vMailPlanTestCatDict) = stattools.categorical(vMailPlanTest, drop=True, dictnames=True)
(oldChurnTestCat, oldChurnTestCatDict) = stattools.categorical(oldChurnTest, drop=True, dictnames=True)

#turn these lists into pd.DataFrames
intlPlanTestCatPD = pd.DataFrame(intlPlanTestCat,columns=['intlPlanNo','intlPlanYes'])
vMailPlanTestCatPD = pd.DataFrame(vMailPlanTestCat,columns=['vMailPlanNo','vMailPlanYes'])
oldChurnTestCatPD = pd.DataFrame(oldChurnTestCat,columns=['oldChurnFalse','oldChurnTrue'])

#drop categorical vals from train and test data
#we pass toDrop List here that we created before we made the dummy vals, additionally we want that performed in place
X.drop(toDrop, axis=1, inplace=True)
XTest.drop(toDrop, axis=1, inplace=True)

#now to add them back in dummy form to the train and test data
X.append(intlPlanCatPD, ignore_index=True, sort=False)#pakage version dependent on whether to sort or not, so since I want to call this function 6 times lets not sort
X.append(vMailPlanCatPD, ignore_index=True, sort=False)
X.append(oldChurnCatPD, ignore_index=True, sort=False)
XTest.append(intlPlanTestCatPD, ignore_index=True, sort=False)
XTest.append(vMailPlanTestCatPD, ignore_index=True, sort=False)
XTest.append(oldChurnTestCatPD, ignore_index=True, sort=False)

#grab column names 
xNames = X.columns.values
yNames = ['True','False']

start = time.time()
#create cart models and export their graphs
cart01 = DecisionTreeClassifier(criterion="gini",max_leaf_nodes=2).fit(X,y)#criterion="gini" results in CART
#export_graphviz(cart01, out_file='cart01.dot', feature_names=xNames, class_names=yNames)
cart02 = DecisionTreeClassifier(criterion="gini",max_leaf_nodes=5).fit(X,y)
#export_graphviz(cart02, out_file='cart02.dot', feature_names=xNames, class_names=yNames)
cart03 = DecisionTreeClassifier(criterion="gini",max_leaf_nodes=10).fit(X,y)
#export_graphviz(cart02, out_file='cart03.dot', feature_names=xNames, class_names=yNames)
#commented out exports since im including the dot files
#only difference is max_leaf_nodes
end = time.time()
ellapsed = end-start#state of the art benchmarking tool!
print('Creating three CART models took ',ellapsed,' seconds.')

start = time.time()
#lets collect our predictions
predCart01 = cart01.predict(XTest)
predCart02 = cart02.predict(XTest)
predCart03 = cart03.predict(XTest)
end = time.time()
ellapsed = end-start#state of the art benchmarking tool!
print('Predictions on three CART models took ',ellapsed,' seconds.')

#now lets make confusion matrices and accs
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
start = time.time()
#create C5.0 models and export their graphs
c50_01 = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=2).fit(X,y)#criterion="entropy" results in C5.0
#export_graphviz(cart01, out_file='c50_01.dot', feature_names=xNames, class_names=yNames)
c50_02 = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=5).fit(X,y)
#export_graphviz(cart02, out_file='c50_02.dot', feature_names=xNames, class_names=yNames)
c50_03 = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=10).fit(X,y)
#export_graphviz(cart02, out_file='c50_03.dot', feature_names=xNames, class_names=yNames)
#only difference is max_leaf_nodes
end = time.time()
ellapsed = end-start#state of the art benchmarking tool!
print('Creating three C5.0 models took ',ellapsed,' seconds.')

start = time.time()
#lets collect our predictions
predc50_01 = c50_01.predict(XTest)
predc50_02 = c50_02.predict(XTest)
predc50_03 = c50_03.predict(XTest)
end = time.time()
ellapsed = end-start#state of the art benchmarking tool!
print('Predictions on three C5.0 models took ',ellapsed,' seconds.')

#now lets make confusion matrices and accs
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

#C5.0 appears to have similar accuracy values compared to CART, but C5.0 obtains greater 
# accuracies than CART when the max_leaf_nodes variable is increased. There is a difference in 
# time to create CART vs C5.0 models with C5.0 models taking slightly longer. However, there is 
# no tangible difference in prediction times.