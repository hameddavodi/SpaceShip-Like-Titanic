## File and Data Field Descriptions

  - train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
  - PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is  their number within the group. People in a group are often family members, but not always.
  - HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
  - CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
  - Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
  - Destination - The planet the passenger will be debarking to.
  - Age - The age of the passenger.
  - VIP - Whether the passenger has paid for special VIP service during the voyage.
  - RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
  - Name - The first and last names of the passenger.
  - Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
  - test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.
  - sample_submission.csv - A submission file in the correct format.
  - PassengerId - Id for each passenger in the test set.
  - Transported - The target. For each passenger, predict either True or False.
```python
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
import warnings
warnings.simplefilter(action='ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
```
This is the import section of the code. It imports several libraries including NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn, Warnings, Logistic Regression, and Recursive Feature Elimination. These libraries will be used in the code for various purposes like data preprocessing, visualization, feature selection, and machine learning.

```python
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df=train_df.drop("Name",axis=1)
test_df=test_df.drop("Name",axis=1)
```
These lines of code read the training and test datasets from CSV files and store them in the variables train_df and test_df, respectively. The drop() method is used to remove the 'Name' column from both datasets, as it is not needed for the analysis.

```python
from sklearn.preprocessing import LabelEncoder
# integer encode
label_encoder = LabelEncoder()
train_df["VIP"]=label_encoder.fit_transform(train_df["VIP"])
train_df["HomePlanet"]=label_encoder.fit_transform(train_df["HomePlanet"])
train_df["Destination"]=label_encoder.fit_transform(train_df["Destination"])
train_df["Transported"]=label_encoder.fit_transform(train_df["Transported"])
train_df["CryoSleep"]=label_encoder.fit_transform(train_df["CryoSleep"])
test_df["CryoSleep"]=label_encoder.fit_transform(test_df["CryoSleep"])
test_df["VIP"]=label_encoder.fit_transform(test_df["VIP"])
test_df["HomePlanet"]=label_encoder.fit_transform(test_df["HomePlanet"])
test_df["Destination"]=label_encoder.fit_transform(test_df["Destination"])
```
These lines of code use the LabelEncoder() function from Scikit-Learn to encode categorical variables as numerical values. The fit_transform() method is used to fit and transform the categorical columns in train_df and test_df.

```python
try:
    new = train_df["Cabin"].str.split("/",expand = True)
    train_df["Deck"]= new[0]
    train_df["Num"]= new[1]
    train_df["side"]= new[2]
    train_df.drop(columns =["Cabin"], inplace = True)

except:
    pass
try: 
    new = test_df["Cabin"].str.split("/", expand = True)
    test_df["Deck"]= new[0]
    test_df["Num"]= new[1]
    test_df["side"]= new[2]
    test_df.drop(columns =["Cabin"], inplace = True)

except:
    pass
```
These lines of code split the 'Cabin' column into three separate columns ('Deck', 'Num', 'side') by using the str.split() method on the '/' separator. The expand=True argument is used to split the column into separate columns. The 'Cabin' column is then dropped from both datasets using the drop() method.

```python 
train_df["Deck"]=label_encoder.fit_transform(train_df["Deck"])
train_df["side"]=label_encoder.fit_transform(train_df["side"])
test_df["Deck"]=label_encoder.fit_transform(test_df["Deck"])
test_df["side"]=label_encoder.fit_transform(test_df["side"])

```
This block of code is performing encoding on two categorical features called "Deck" and "side" present in both the training and testing datasets. The label_encoder object is used to convert these categorical features into numerical values.

fit_transform method of the label_encoder object is used to fit the encoder to the training data and then transform it. The same encoder is then used to transform the corresponding features in the testing dataset.

### Correlation Heatmap

```python

Selected_features = [
    
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Deck', 'Num', 'side'
    
]
X = train_df[Selected_features]

plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()
```
This block of code is selecting a subset of features from the train_df dataframe and assigning it to the variable X. The selected features are:

Next, this code is creating a heatmap using sns.heatmap() function from the seaborn library. This heatmap shows the correlation between each pair of features in the X DataFrame.

The annot=True parameter adds the correlation coefficient value on each cell of the heatmap. The cmap="RdYlGn" parameter sets the color map for the heatmap.

<img width="712" alt="Screenshot 2023-04-19 at 10 55 28" src="https://user-images.githubusercontent.com/109058050/233023204-90d8b3e6-c3a1-4298-b04e-0c94e9075d94.png">

### Data Normalization


```python
x = train_df #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
training = pd.DataFrame(x_scaled)

y = test_df #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
testing = pd.DataFrame(y_scaled)
```
These lines of code are used to perform Min-Max normalization on the training and testing data.

First, the training data is stored in a pandas DataFrame named train_df. Then, an instance of the MinMaxScaler class from the preprocessing module of scikit-learn is created and stored in the min_max_scaler variable. The fit_transform() method of the min_max_scaler object is then called on the x array (which represents the training data), which performs the normalization and returns a new numpy array, which is stored in the x_scaled variable. Finally, the x_scaled array is converted back into a pandas DataFrame named training.

The same process is repeated for the testing data. The testing data is stored in a numpy array named test_df. Another instance of the MinMaxScaler class is created and stored in the min_max_scaler variable. The fit_transform() method of this instance is then called on the y array (which represents the testing data) to normalize it, and the resulting array is stored in the y_scaled variable. Finally, the y_scaled array is converted back into a pandas DataFrame named testing.

Overall, this code performs Min-Max normalization on the data and stores the normalized data in two separate pandas DataFrames, one for training data (training) and one for testing data (testing).

Still a bit of adjustments:
```python
try:
    training.columns=[
    'PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Transported', 'Deck', 'Num', 'side'
]
except:
    training.drop(["PassengerId"],axis=1,inplace=True)
finally:
    training.columns=[
    'PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Transported', 'Deck', 'Num', 'side'
    ]
    training["PassengerId"]=train_df['PassengerId']
try:        
    testing.columns=[
    'PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Deck', 'Num', 'side'
]
except:
    testing.drop(["PassengerId"],axis=1,inplace=True)
finally:
    testing.columns=[
    'PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Deck', 'Num', 'side']
    testing["PassengerId"]=test_df["PassengerId"]
```
This code renames the columns of the training and testing DataFrames.

First, it tries to rename the columns of the "training" DataFrame to the specified column names using a try-except block. If the column renaming fails, it drops the "PassengerId" column from the "training" DataFrame and then renames the remaining columns using a finally block.

Next, the code renames the columns of the "testing" DataFrame in a similar manner. It first tries to rename the columns using the specified column names, and if that fails, it drops the "PassengerId" column from the "testing" DataFrame before renaming the remaining columns using a finally block.

Also, dropping NaN values:
```python 
training.dropna(inplace=True)
testing.dropna(inplace=True)
```
### Feature Selection and Feature Tunning

1- Chi-Squared Test

```python 
from scipy.stats import chi2_contingency


# Define the target variable
target = 'Transported'

# Define the list of features
features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Deck', 'Num', 'side'
]

# Create a contingency table for each feature
for feature in features:
    contingency_table = pd.crosstab(training[feature], training[target])
    
    # Calculate the Chi-Squared statistic and p-value
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Print the results
    print(f"Feature: {feature}")
    print(f"Chi-Squared: {chi2}")
    print(f"P-value: {p_value}")
    print("-" * 30)
```
This code performs a chi-squared test of independence on each feature in the features list with respect to the target variable Transported.

First, the code imports the chi2_contingency function from the scipy.stats module.

Next, it defines the target variable as 'Transported' and creates a list of features named features.

For each feature in the features list, the code generates a contingency table using the pd.crosstab() function with the feature and target variable as inputs.

Then, the chi-squared test is performed using the chi2_contingency() function, which takes the contingency table as input and returns the chi-squared statistic, p-value, degrees of freedom, and expected frequencies. These values are stored in chi2, p_value, dof, and expected, respectively.

```
Feature: HomePlanet
Chi-Squared: 268.63487698174964
P-value: 6.09265575043053e-58
------------------------------
Feature: CryoSleep
Chi-Squared: 1551.8886069991156
P-value: 0.0
------------------------------
Feature: Destination
Chi-Squared: 108.22289914296638
P-value: 2.647003128327146e-23
------------------------------
Feature: Age
Chi-Squared: 228.04080321314734
P-value: 2.181935267370355e-16
------------------------------
Feature: VIP
Chi-Squared: 11.532517932796942
P-value: 0.0031314504901248357
------------------------------
Feature: RoomService
Chi-Squared: 1726.0371275018924
P-value: 5.71001275732341e-23
------------------------------
Feature: FoodCourt
Chi-Squared: 1859.7508431930924
P-value: 5.894924561819268e-16
------------------------------
Feature: ShoppingMall
Chi-Squared: 1588.0015375189298
P-value: 1.0273508819395727e-25
------------------------------
Feature: Spa
Chi-Squared: 1764.2583161615216
P-value: 9.41720901097681e-22
------------------------------
Feature: VRDeck
Chi-Squared: 1627.9149637923595
P-value: 5.318402354621806e-16
------------------------------
Feature: Deck
Chi-Squared: 343.01240604808106
P-value: 3.8571400858314705e-70
------------------------------
Feature: Num
Chi-Squared: 1891.3395907185616
P-value: 0.028323616346795522
------------------------------
Feature: side
Chi-Squared: 88.73834339045419
P-value: 4.506450894721151e-21
------------------------------
```

2- RFE

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# Define the target variable
target = 'Transported'

# Define the list of features
features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Deck', 'Num', 'side'
]

# Create a Logistic Regression model
model = LogisticRegression()

# Create an RFE object with the model and number of features to select
rfe = RFE(model, n_features_to_select=1)

# Fit the RFE object to the data and get the ranking of features
rfe.fit(training[features], training[target])
ranked_features = rfe.ranking_

# Print the ranking of features
for i, feature in enumerate(features):
    print(f"Rank {ranked_features[i]}: {feature}")
```
This code uses Recursive Feature Elimination (RFE) with a logistic regression model to rank the importance of each feature in predicting the target variable Transported.

First, the code imports the RFE class from the sklearn.feature_selection module and the LogisticRegression class from the sklearn.linear_model module.

Next, it defines the target variable as 'Transported' and creates a list of features named features.

Then, the code creates a logistic regression model using the LogisticRegression() function.

After that, it creates an RFE object using the RFE() function, which takes the created logistic regression model and the number of features to select as input.

The fit() method is then called on the RFE object using the training data training[features] and the target variable training[target] as inputs. The fit() method fits the model to the data and selects the most important features based on their ranking.

```
Rank 11: HomePlanet
Rank 6: CryoSleep
Rank 10: Destination
Rank 9: Age
Rank 13: VIP
Rank 1: RoomService
Rank 4: FoodCourt
Rank 5: ShoppingMall
Rank 2: Spa
Rank 3: VRDeck
Rank 7: Deck
Rank 12: Num
Rank 8: side
```

Then we select the Features based on the overall conclusion of these tests. 
```python
Ranking_features=['RoomService','Spa','VRDeck','FoodCourt','ShoppingMall','CryoSleep','Deck','side','Age']
```
### Logestic Regression

```python 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import accuracy_score


# create X (features) and y (response)
X = training[Ranking_features]
y = training['Transported']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.98)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```
Results:
```
Using a threshold of 0.204 guarantees a sensitivity of 0.980 and a specificity of 0.184, i.e. a false positive rate of 81.61%.
Accuracy: 0.7635997313633311
```
### Random Forest

```python 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X = training[Ranking_features]
y = training['Transported']


# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=20
)

# check classification scores of logistic regression
logreg = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
X_test = testing[Ranking_features]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.98)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

```
Results:
```
Using a threshold of 0.188 guarantees a sensitivity of 0.981 and a specificity of 0.289, i.e. a false positive rate of 71.12%.
Accuracy: 0.8093959731543624
```
Obviously, the Random Forest performed better than Logistic Regression.
