import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq

signal = np.genfromtxt("signal.dat",delimiter=" , ")

incompletos = np.genfromtxt("incompletos.dat",delimiter=",")

sx = signal[:,0]
sy = signal[:,1]

ix = incompletos[:,0]
iy = incompletos[:,1]


""""
plt.plot(sx,sy)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Signal")
plt.savefig("CorrealSergio_signal.pdf")
"""

def FT(f):
	N = f.shape[0]
	Y = np.zeros(N, dtype = complex)
	Z = np.exp(-1j*2*np.pi/N)
	for n in range(N):
		r=0
		for k in range(N):
			r += f[k] * Z**(n*k)
		Y[n] = r
	return Y

print "La imprementacion de freq es propia"

def darFreq(np,dtp):
	r = []
	if(np%2==0): # Si np es par
		i = 0
		j = -(np-1)/2
		while(i<np/2):
			r.append(i)
			i += 1
		while(j<0):
			r.append(j)
			j += 1
	else: # Si np es impar
		i = 0
		j = -np/2
		while(i< ((np-1)/2 +1)):
			r.append(i)
			i += 1
		while(j<0):
			r.append(j)
			j += 1
	return r
"""
nu = 128 # number of point in the whole interval
f = 200.0 #  frequency in Hz
dt = 1 / (f * 32 ) #32 samples per unit frequency
t = np.linspace( 0, (nu-1)*dt, nu)

y = np.cos(2 * np.pi * 200 * t) - 0.4 * np.sin(2 * np.pi * (2*200) * t )
"""

nu = signal.shape[0]
dt = sx[1]- sx[0]

FT_y = FT(sy) / nu # FT normalizada

freqmia = darFreq(nu,dt)
plt.plot(freqmia,abs(FT_y))
plt.xlabel("freq(Hz)")
plt.ylabel("f(w)")
plt.title("signal FT")
plt.show()

#plt.plot(ix,iy)
#plt.xlabel("t")
#plt.ylabel("y")
#plt.title("Incompleto")
#plt.show()
#plt.savefig("CorrealSergio_incompleto.pdf")