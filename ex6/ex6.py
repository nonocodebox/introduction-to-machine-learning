import numpy as np

def perceptron(data, labels):
	num_samples, image_size, = np.shape(data);
	w = np.zeros(image_size);

	for i in range(num_samples):
		if (labels[i] * np.dot(w, data[i])) <= 0:
			add_vector = np.multiply(labels[i] * np.ones(image_size), data[i])
			w = np.add(w, add_vector)
	return w

def zeroOneLoss(data, labels, w):
	num_samples, image_size, = data.shape
	#Count the errors of the given coordinate. Store the number in 'num_errors'

        num_errors = 0
	for i in range(num_samples):
		sgn = lambda x: -1 if x <= 0 else 1
		if labels[i] != sgn(np.dot(w, data[i])):
			num_errors += 1.0

	zero_one_loss = num_errors/num_samples
	return zero_one_loss

if __name__ == '__main__':
	Xtrain = np.loadtxt("Xtrain");
	Ytrain = np.loadtxt("Ytrain");
	Xtest = np.loadtxt("Xtest");
	Ytest = np.loadtxt("Ytest");

	w = perceptron(Xtrain, Ytrain)
	print 'Zero one loss:', zeroOneLoss(Xtest, Ytest, w)


