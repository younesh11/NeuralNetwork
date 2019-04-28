import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def plot_decision_boundary(X, y_cat, model):
	
	x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
	y_span = np.linspace(min(X[:, 1]) - -0.25, max(X[:, 1]) + 0.25)
	xx, yy = np.meshgrid(x_span, y_span)
	xx_, yy_ = xx.ravel(), yy.ravel()
	grid  = np.c_[xx_, yy_]
	pred_func = model.predict_classes(grid)
	z = pred_func.reshape(xx.shape)
	plt.contourf(xx, yy, z)


n_pts = 500
centers = [[-1, 1], [-1, -1], [1, -1]]
X, Y =datasets.make_blobs(n_samples = n_pts, random_state = 123, centers = centers, cluster_std = 0.4)

#print(X, Y)

y_cat = to_categorical(Y, 3)

model = Sequential()
model.add(Dense(units = 3, input_shape = (2,), activation = 'softmax'))
model.compile(Adam(0.1), loss = 'categorical_crossentropy')

model.fit(x = X, y = y_cat, verbose = 1, batch_size = 50, epochs = 50)

plot_decision_boundary(X, y_cat, model)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1])
plt.scatter(X[Y == 1, 0], X[Y == 1, 1])
plt.scatter(X[Y == 2, 0], X[Y == 2, 1])
plt.show()
