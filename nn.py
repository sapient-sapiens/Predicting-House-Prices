#Data Loading and Pre-Processing 

from sklearn.datasets import load_boston 
import pandas as pd 

boston = load_boston() 
data = pd.DataFrame(boston.data, columns = boston.feature_names)
data['PRICE'] = boston.target 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

X = data.drop('PRICE', axis = 1)
y = data['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 
