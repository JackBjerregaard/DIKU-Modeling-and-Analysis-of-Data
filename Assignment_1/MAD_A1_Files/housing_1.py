import matplotlib.pyplot as plt
import numpy

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

# (a) compute mean of prices on training set

mean = numpy.mean(t_train)
print(f"a) Mean house price on training set: ${mean:.2f}k")

# (b) RMSE function
def rmse(t, tp):
    return numpy.sqrt(numpy.mean((t-tp)**2)) # begin by finding differences, square the errors, and take the mean of the squared errors and finnaly take the squre root

# Create predictions (all equal to mean)
predictions = numpy.full_like(t_test, mean)

# Compute RMSE
test_rmse = rmse(t_test, predictions)
print(f"b) RMSE on test set: ${test_rmse:.2f}k")

# (c) visualization of results
plt.figure()
plt.scatter(t_test, predictions)
plt.xlabel("True price")
plt.ylabel("Predicted price")
plt.title("Mean Model")
plt.savefig("housing_1_scatter.png")



