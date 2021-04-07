import numpy as np

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
	pass

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
	#print "\n\n#1", faces[0], faces.shape, "\n\n", faces[0][47][47];


	arr = faces.flatten();
	#print "\n\n#2", arr, arr.shape, arr[48*48 - 1];

	arr = np.reshape(faces, (faces.shape[0], faces.shape[1] * faces.shape[2]));
	#print "\n\n#3", arr, arr.shape, "\n\n";

	arr = arr.transpose();
	#print "\n\n#4", arr, arr.shape, "\n\n";

	appended_arr = np.atleast_2d(np.ones(faces.shape[0], dtype=int));
	#print appended_arr;

	arr = np.vstack((arr, appended_arr));
	#print arr;

	return arr;

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):

	
	bias = w[-1];
	w_fit = w[0:-1];

	index = 0;
	sum = 0;
	for i in Xtilde.T:
		guess = i[0:-1].T.dot(w_fit) + bias;

		sum = sum + (guess - y[index]) **2
		index = index + 1;


	return sum / (2 * Xtilde.shape[1] - 1);



# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
	error = (1 / float(Xtilde.shape[1])) * (y - Xtilde.T.dot(w)).T.dot(y - Xtilde.T.dot(w));
	
	if alpha != 0:
		regulation = alpha / (2 * Xtilde.shape[1]);
		regulation = regulation * w.T.dot(w);
	else:
		regulation = 0;

	return error + regulation;

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
	return np.linalg.solve(Xtilde.dot(Xtilde.T), Xtilde.dot(y));

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
	return gradientDescent(Xtilde, y);

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
	ALPHA = 0.1
	return gradientDescent(Xtilde, y, ALPHA);

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
	EPSILON = 0.003  # Step size aka learning rate
	T = 5000  # Number of gradient descent iterations
	
	w = 0.01 * np.random.randn(Xtilde.shape[0]);


	for i in range(0, T):
		guess = Xtilde.T.dot(w);
		w = w - EPSILON * (1 / float(Xtilde.shape[1])) * (Xtilde.dot(guess - y));
		
	return w;
if __name__ == "__main__":
# Load data

	Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
	ytr = np.load("age_regression_ytr.npy")
	Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
	yte = np.load("age_regression_yte.npy")

	#print np.load("age_regression_Xtr.npy")[0], Xtilde_tr, Xtilde_tr.shape;

	w1 = method1(Xtilde_tr, ytr)
	w2 = method2(Xtilde_tr, ytr)
	w3 = method3(Xtilde_tr, ytr)

	print fMSE(w1, Xtilde_te, yte);
	print gradfMSE(w2, Xtilde_te, yte)
	print gradfMSE(w3, Xtilde_te, yte)
# Report fMSE cost using each of the three learned weight vectors
# ...
