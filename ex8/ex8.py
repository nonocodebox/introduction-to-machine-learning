import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib

def trainClass(x, y, degree):
	X = getPolyMatrix(x, degree)
	A = np.dot(X, X.T)
	APlus = np.linalg.pinv(A)

	return np.dot(np.dot(APlus, X), y)

def train(x, y):
	hypo_list = []
	for i in range(1,16):
		hypo_list.append(trainClass(x, y, i))

	# Plot train hypotesis error
	plot(x, y, hypo_list)

	return hypo_list

def validate(x, y, hypo_list):
	h_min = hypo_list[0]
	min_error = lossFunction(x, y, h_min)

	for i in range(1,15):
		error = lossFunction(x, y, hypo_list[i])
		if min_error > error:
			h_min = hypo_list[i]
			min_error = error

	# Plot validation hypotesis error
	plot(x, y, hypo_list)

	return h_min

def test(x, y, h):
	return lossFunction(x, y, h)

def lossFunction(x, y, w):
	num_samples, = np.shape(x)
	degree = len(w) - 1
	X = getPolyMatrix(x, degree)
	sum = 0
	for i in range(num_samples):
		sum += (np.dot(w, X[:,i]) - y[i]) ** 2
	return sum / num_samples

def getPolyMatrix(x, degree):
	num_samples, = np.shape(x)
	polys = np.zeros([degree + 1, num_samples])
	for i in range(num_samples):
		for j in range(degree + 1):
			polys[j, i] = x[i] ** j
	return polys

def plot(x, y, hypo_list):
	# Plot hypotesis error
	plt.xlim(1,15)
	plt.xlabel('Polynomial degree')
	plt.ylabel('Error value')
	plt.plot(range(1, 16), map(lambda i: lossFunction(x, y, hypo_list[i]), range(0,15)))

if __name__ == '__main__':
	x = np.loadtxt("X.txt")
	y = np.loadtxt("Y.txt")

	train_set = x[0:20]
	validation_set = x[20:121]
	test_set = x[121:221]

	train_labels = y[0:20]
	validation_labels = y[20:121]
	test_labels = y[121:221]

	hypo_set = train(train_set, train_labels)
	h = validate(validation_set, validation_labels, hypo_set)

	print "Test error:", test(test_set, test_labels, h)
	print "Polynomial's degree:", len(h)-1

	plt.title('Hypothesis errors vs. Polynomial degreee')
	plt.legend(('Train hypotesis error', 'Validation hypotesis error'))
	plt.show()
