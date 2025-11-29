import numpy
import pandas as pd
import linweighreg
import matplotlib.pyplot as plt

numpy.set_printoptions(precision=3) #for .3f for arrays

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (b) fit weighted linear regression using all features with alpha_n = t_n^2
alpha = t_train ** 2

model = linweighreg.WeightedLinearRegression()
model.fit(X_train, t_train, alpha)

print(f"\nWeighted regression coefficients:")
print(model.w)

# evaluation of results
predictions = model.predict(X_test)

#RMSE
rmse = numpy.sqrt(numpy.mean((predictions-t_test)**2))
print(f"\nRMSE: {rmse:.3f}")

plt.figure()
plt.scatter(t_test, predictions)
plt.xlabel("True price")
plt.ylabel("Predicted price")
plt.title("Weighted Linear Regression (alpha = t^2)")
plt.savefig("scatter_weighted.png")

plt.show()
