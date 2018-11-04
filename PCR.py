import subprocess
import matplotlib.pylab as plt
import numpy as np

#subprocess.call("wget http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat", shell=True)

data = np.genfromtxt("WDBC.dat", delimiter=",",dtype = (float,str,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float), names = "a")
print data.shape
print data[0,1]

# dtype = (float,str,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float))
# ,names = "Q, W, E, R, T, Y, U, I, O, P, A, S, D, F, G, H, J, K, L, Z, X, C, V, B, N, M, Q, W, ER, RT, TY, YU")
