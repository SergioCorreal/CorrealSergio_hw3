import numpy as np
import matplotlib.pylab as plt

signal = np.genfromtxt("signal.dat",delimiter=" , ")

f = signal.shape[0]
incompletos = np.genfromtxt("incompletos.dat",delimiter=" , ")

sx = signal[:,0]
sy = signal[:,1]

ix = incompletos[0]
iy = incompletos[1]

plt.plot(sx,sy)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Signal")
plt.savefig("CorrealSergio_signal.pdf")

plt.plot(ix,iy)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Incompleto")
plt.savefig("CorrealSergio_incompleto.pdf")