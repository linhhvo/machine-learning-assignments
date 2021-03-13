import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from dnn_solution import Layer, DNNClassifier
from utils_loaddata import load_dataset

train_x_orig, y_train, test_x_orig, y_test, classes = load_dataset()
print ("The shape of the original X_train: " + str(train_x_orig.shape))
print ("The shape of the original y_train:", y_train.shape)

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
X_train = train_x_flatten / 255.
X_test = test_x_flatten / 255.
print ("The shape of X_train: " + str(X_train.shape))
print ("The shape of y_train:", y_train.shape)

dnnclf = DNNClassifier()
dnnclf.add(Layer(20, 'relu'))
dnnclf.add(Layer(7, 'relu'))
dnnclf.add(Layer(5, 'relu'))
dnnclf.add(Layer(1, 'sigmoid'))

eta = 0.0075
iterations = 2500
dnnclf.fit(X_train, y_train, learning_rate=eta, num_epochs=iterations, print_cost=True)

y_pred = dnnclf.predict(X_train)
print("Training accuracy: %.2f" % (accuracy_score(y_train.T, y_pred.T)*100) + '%')
y_pred = dnnclf.predict(X_test)
print("Testing accuracy: %.2f" % (accuracy_score(y_test.T, y_pred.T)*100) + '%')

# plot the cost
plt.plot(np.squeeze(dnnclf.costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(eta))
plt.show()
