import numpy as np
import math as m
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('dane15.txt')

x = a[:,[0]]    #INPUT
y = a[:,[1]]    #OUTPUT

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)

x=X_train
y=y_train

c = np.hstack([x, np.ones(x.shape)])    # model y = ax+b {a,b}
v = np.linalg.pinv(c) @ y
e = sum((y-(v[0]*x + v[1]))**2)/len(x)
#print(c)
#print(v)
print(e)

c1 = np.hstack([x**2,x, np.ones(x.shape)]) # model y = ax2 + bx + c
v1 = np.linalg.pinv(c1) @ y
e1 = sum((y-(v1[0]*x**2 + v1[1]*x + v1[2]))**2)/len(x)
#print(c1)
#print(v1)
print(e1)


plt.plot(x,y,'g^')
plt.plot(X_train, y_train, 'ro')
plt.plot(x, v[0]*x + v[1])
plt.plot(x, v1[0]*x**2 + v1[1]*x + v1[2])

plt.show()
