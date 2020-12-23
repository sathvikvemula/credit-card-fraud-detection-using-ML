# credit-card-fraud-detection-using-ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('sample_data/creditcard.csv')
fraud=df[df['Class']==1]
normal=df[df['Class']==0]
from sklearn.model_selection import train_test_split
x = df.drop('Class', axis = 1)
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 1)
from xgboost import XGBClassifier
model = XGBClassifier()
# fit the model with the training data
model.fit(x_train,y_train)
predict_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)
