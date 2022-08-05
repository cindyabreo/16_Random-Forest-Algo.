import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  

#Step 1:

url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url, header=0, sep=",")

#Step 2:
df.head(5)
df.info()
df.describe()
df.sample(5)
X = df.drop(['PassengerId','Survived','Pclass','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)
									
y = df['Survived']
X.head(5)
y.head(5)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#Train the model¶
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
clf

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

from sklearn import tree

from matplotlib import pyplot as plt

plt.figure(figsize=(10,8))
tree.plot_tree(clf)
plt.show()