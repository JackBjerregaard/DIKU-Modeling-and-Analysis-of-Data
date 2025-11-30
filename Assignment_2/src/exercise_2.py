import matplotlib.pyplot as plt
import numpy

# Note that i do not use any imports for weighted lin reg as i wanted to make this file all
# encompassing for this question, thus i have defined functions in this file
# tldr: this python file and linreg are independant of each other
numpy.set_printoptions(precision=10)

raw = numpy.genfromtxt("men-olympics-100.txt", delimiter=" ")

# extract columns
x = raw[:, 0]  # years
t = raw[:, 1]  # first place times

t = t.reshape((len(t), 1))


def make_polynomial(x, degree):
    N = len(x)
    X = numpy.zeros((N, degree + 1))

    for i in range(degree + 1):
        X[:, i] = x**i

    return X


def reg_lin_reg(X, t, lambdaVal):
    D = X.shape[1]
    A = X.T @ X + lambdaVal * numpy.eye(D)
    b = X.T @ t
    return numpy.linalg.solve(A, b)


def predict(X, w):
    return X @ w


def loocv(X, t, lambdaVal):
    N = X.shape[0]
    errors = []
    for i in range(N):
        X_train = numpy.delete(X, i, axis=0)
        t_train = numpy.delete(t, i, axis=0)
        X_test = X[i : i + 1, :]
        t_test = t[i]
        prediction = predict(X_test, reg_lin_reg(X_train, t_train, lambdaVal))
        errors.append((prediction[0, 0] - t_test[0]) ** 2)
    return numpy.mean(errors)

# a)
lambdaVals = numpy.logspace(-8, 0, 100, base=10)

X = make_polynomial(x, 1)  # design matrix for first order poly

cv_errors = []
for lam in lambdaVals:
    cv_errors.append(loocv(X, t, lam))

cv_errors = numpy.array(cv_errors)

best_lambda = lambdaVals[numpy.argmin(cv_errors)]

# weights for lambda = 0 (no reg)
w_no_reg = reg_lin_reg(X, t, lambdaVal=0)

# best weights for best lambda
w_best = reg_lin_reg(X, t, lambdaVal=best_lambda)

print(f"a) Best lambda: {best_lambda:.10f}")
print(f"   w(lambda=0): {w_no_reg.flatten()}")
print(f"   w(lambda={best_lambda:.10f}): {w_best.flatten()}")

# plot CV error vs lambda
plt.figure()
plt.semilogx(lambdaVals, cv_errors)
plt.xlabel("Lambda")
plt.ylabel("Leave-One-Out CV Error")
plt.title("CV Error vs Lambda (First-Order Polynomial)")
plt.savefig("exercise2_part_a.png")
plt.show()

# b)
degree = 4
X = make_polynomial(x, degree)  # design matrix: [1, x, x^2, x^3, x^4]

cv_errors_4 = []
for lam in lambdaVals:
    cv_errors_4.append(loocv(X, t, lam))

cv_errors_4 = numpy.array(cv_errors_4)

best_lambda_4 = lambdaVals[numpy.argmin(cv_errors_4)]

# weights for lambda = 0 (no reg)
w_no_reg_4 = reg_lin_reg(X, t, lambdaVal=0)

# best weights for best lambda
w_best_4 = reg_lin_reg(X, t, lambdaVal=best_lambda_4)

print(f"b) Best lambda: {best_lambda_4:.10f}")
print(f"   w(lambda=0): {w_no_reg_4.flatten()}")
print(f"   w(lambda={best_lambda_4:.10f}): {w_best_4.flatten()}")

# plot CV error vs lambda
plt.figure()
plt.semilogx(lambdaVals, cv_errors_4)
plt.xlabel("Lambda")
plt.ylabel("Leave-One-Out CV Error")
plt.title("CV Error vs Lambda (Fourth-Order Polynomial)")
plt.savefig("exercise2_part_b.png")
plt.show()
