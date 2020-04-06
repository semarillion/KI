from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)

mlp=MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10],activation="tanh").fit(X_train,y_train)
mlp_relu=MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10]).fit(X_train,y_train)
mlp_relu_2hl=MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)

mglearn.plots.plot_2d_classification(mlp,X_train,fill=True,alpha=0.3)
mglearn.plots.plot_2d_classification(mlp_relu,X_train,fill=True,alpha=0.3)
mglearn.plots.plot_2d_classification(mlp_relu_2hl,X_train,fill=True,alpha=0.3)


mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Merkmal 0")
plt.ylabel("Merkmal 1")
plt.show()
