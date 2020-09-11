import numpy as np, preprocess, os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

x_train, y_train = preprocess.generate_training_set()

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.15)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

print(x_train.shape, y_train.shape)

# linear: 0.5604395604395604
# poly: 0.5457875457875457
# rbf: 0.5860805860805861
# sigmoid: 0.4981684981684982
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_train, y_train)

print(np.sum((y_test == clf.predict(x_test))))
print((y_test == clf.predict(x_test)).shape, y_test.shape)

print(np.sum((y_test == clf.predict(x_test))) / y_test.shape[0])