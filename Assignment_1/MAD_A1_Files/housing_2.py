import numpy
import pandas as pd
import linreg
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

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train) #  builds design matrix we = etc,

print(f"intercept: {model_single.w[0,0]:.3f}")
print(f"CRIM: {model_single.w[1,0]:.3f}")

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train) 

print(f"All weights:\n{model_all.w}")

# (d) evaluation of results

single_prediction = model_single.predict(X_test[:,0])
all_prediction = model_all.predict(X_test)

#RMSE 
single_rmse = numpy.sqrt(numpy.mean((single_prediction-t_test)**2))
all_rmse = numpy.sqrt(numpy.mean((all_prediction-t_test)**2))
print(f"RMSE single: {single_rmse:.3f}")
print(f"RMSE all: {all_rmse:.3f}")

plt.figure()
plt.scatter(t_test, single_prediction)
plt.xlabel("True price")
plt.ylabel("Predicted price")
plt.title("Single feature (CRIM)")
plt.savefig("scatter_single.png")

plt.figure()
plt.scatter(t_test, all_prediction)
plt.xlabel("True price")
plt.ylabel("Predicted price")
plt.title("All features")
plt.savefig("scatter_all.png")

plt.show()

