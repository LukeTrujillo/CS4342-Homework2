import numpy as np
from matplotlib import pyplot as plt

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

	guess = Xtilde[0:-1].T.dot(w[0:-1]) + w[-1];

	if alpha != 0:
		regulation = alpha / (2 * (Xtilde.shape[1] - 1));
		regulation = regulation * w[0:-1].T.dot(w[0:-1]);
	else:
		regulation = 0;

	return (1 / float(Xtilde.shape[1])) * (Xtilde.dot(guess - y)) + regulation;

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

def biggestErrors(w, Xtilde, y):
	performance = np.array([]);
	
	bias = w[-1];
	w_fit = w[0:-1];

	index = 0;
	for i in Xtilde.T:
		guess = i[0:-1].T.dot(w_fit) + bias;
		groundTruth = y[index];

		error = ((guess - groundTruth)**2)**0.5

		data = np.array([index, error, guess, groundTruth]);

		performance = np.append(performance, data, axis = 0);

		index = index + 1;

	formatted =  np.reshape(performance, (-1, 4))
	return np.flip(formatted[formatted[:, 1].argsort()], 0)[0:5, :];	

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
	EPSILON = 0.003  # Step size aka learning rate
	T = 5000  # Number of gradient descent iterations
	
	w = 0.01 * np.random.randn(Xtilde.shape[0]);

	for i in range(0, T):
		w = w - EPSILON * gradfMSE(w, Xtilde, y, alpha);
	return w;


def trainPolynomialRegressor(x, y, d):

	designMatrix = np.array([]);

	for i in range(0, d + 1):
		powerComp = x ** i;
		designMatrix = np.append(designMatrix, powerComp, axis = 0);
	
	designMatrix = np.reshape(designMatrix, (d + 1, -1));

	return np.linalg.solve(designMatrix.dot(designMatrix.T), designMatrix.dot(y));

if __name__ == "__main__":

	Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
	ytr = np.load("age_regression_ytr.npy")
	Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
	yte = np.load("age_regression_yte.npy")


	w1 = method1(Xtilde_tr, ytr)
	print "Analtical Solution MSE:", "Training Set = ", fMSE(w1, Xtilde_tr, ytr), "Testing Set = ", fMSE(w1, Xtilde_te, yte);
	
	w2 = method2(Xtilde_tr, ytr)
	print "Gradient Descent MSE:", "Training Set = ", fMSE(w2, Xtilde_tr, ytr), "Testing Set = ", fMSE(w2, Xtilde_te, yte);

	w3 = method3(Xtilde_tr, ytr)
	print "Gradient Descent with Regulation MSE:", "Training Set = ", fMSE(w3, Xtilde_tr, ytr), "Testing Set = ", fMSE(w3, Xtilde_te, yte);

	#top5 = biggestErrors(w3, Xtilde_te, yte);
	#for row in top5:
	#	print row;
	#	img = Xtilde_te.T[int(row[0]), 0:-1];
	#	img = np.reshape(img, (48, 48));
	#	plt.imshow(img)
	#	plt.show();
	
	#w1_image = np.reshape(w1[:-1], (48, 48));
	#plt.imshow(w1_image);
	#plt.show();

	#w2_image = np.reshape(w2[:-1], (48, 48));
	#plt.imshow(w2_image);
	#plt.show();

	#w3_image = np.reshape(w3[:-1], (48, 48));
	#plt.imshow(w3_image);
	#plt.show();
	
