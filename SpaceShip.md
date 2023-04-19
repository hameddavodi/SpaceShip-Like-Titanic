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



