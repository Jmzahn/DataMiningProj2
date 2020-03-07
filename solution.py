import pandas as pd
import numpy as np

churn = pd.read_csv('./churn')
churnVals = churn['Churn'].to_numpy()
numTrue = np.intc(0)
numFalse = np.intc(0)
one = np.intc(1)

for val in churnVals:
    if(val == True):
        numTrue += one
    else:
        numFalse += one

print('True values: ',numTrue,'\nFalse values: ',numFalse)
