import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets
iris = datasets.load_iris()

iris_data = iris.data
iris_data = pd.DataFrame(iris_data,columns=iris.feature_names)
iris_data['class'] = iris.target
iris_data.head()

iris.target_names

print(iris_data.shape)
print(iris_data.describe())

import seaborn as sns
sns.boxplot(data=iris_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(2,5)})
iris_data.hist('class')
iris_data.hist('sepal length (cm)')
iris_data.hist('sepal width (cm)')
iris_data.hist('petal length (cm)')
iris_data.hist('petal width (cm)')

x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()

x_index = 0
y_index = 2

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()

x_index = 0
y_index = 3

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
X= iris_data.values[:,0:4]
Y = iris_data.values[:,4]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=.3,random_state=42)

model = KNeighborsClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))

model = SVC()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))

model = RandomForestClassifier(n_estimators=5)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))

model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))

model = RandomForestClassifier(n_estimators=500)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))

