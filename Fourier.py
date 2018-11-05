import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq
from scipy import interpolate

signal = np.genfromtxt("signal.dat",delimiter=" , ")

incompletos = np.genfromtxt("incompletos.dat",delimiter=",")

sx = signal[:,0]
sy = signal[:,1]

ix = incompletos[:,0]
iy = incompletos[:,1]



CorrealSergio_signal = plt.figure()
plt.plot(sx,sy)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Signal")
CorrealSergio_signal.savefig("CorrealSergio_signal.pdf")

CorrealSergio_incompleto = plt.figure()
plt.plot(ix,iy)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Incompleto")
CorrealSergio_incompleto.savefig("CorrealSergio_incompleto.pdf")


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

print "La implementacion de freq es propia"

def darFreq(npp,dtp): # npp es numero de frecuencias y dt es el espaciamiento del dominio de la funcion original
	r = []
	factor = 1/ (npp*dtp)
	if(npp%2==0): # Si np es par
		i = 0
		j = -(npp-1)/2
		while(i<npp/2):
			r.append(i*factor)
			i += 1
		while(j<0):
			r.append(j*factor)
			j += 1
	else: # Si np es impar
		i = 0
		j = -npp/2
		while(i< ((npp-1)/2 +1)):
			r.append(i*factor)
			i += 1
		while(j<0):
			r.append(j*factor)
			j += 1
	return np.asarray(r)


nu = signal.shape[0] # numero de datos
dt = sx[1] - sx[0]

FT_sy = FT(sy) / nu # FT normalizada

freqmia = darFreq(nu,dt) 

CorrealSergio_TF = plt.figure()
plt.plot(freqmia,abs(FT_sy))
plt.xlabel("freq(Hz)")
plt.ylabel("FT")
plt.title("signal FT")
CorrealSergio_TF.savefig("CorrealSergio_TF.pdf")


index = np.where(abs(FT_sy)>0.6)

frecuencias_principales = freqmia[index]

print "Las frecuencias principales de signal son:\n"
for i in frecuencias_principales:
	if(i>0):
		print i,"\n"

# Limpio senal
FT_sy_f1 = FT_sy.copy()
FT_sy_f1[abs(freqmia) > 1000] = 0

clean_sy_f1 = np.fft.ifft(FT_sy_f1*nu)


CorrealSergio_filtrada = plt.figure()
plt.plot(sx, np.real(clean_sy_f1), color ="red")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Clean signal")
CorrealSergio_filtrada.savefig("CorrealSergio_filtrada.pdf")




print "La transformada de Fourier de incompletos no se puede realizar porque la senal no esta evaluada en intervalos espaciados uniformemente."

def interpola(dat, x_interpol): # dat son los datos y x_interpol son los valores en x donde se quiere saber cuanto vale la funcion interpolada
	x = dat[:,0]
	y = dat[:,1]
	f_cuadratica= interpolate.interp1d(x,y, kind="quadratic")
	f_cubica = interpolate.interp1d(x,y, kind="cubic")
	
	return f_cuadratica(x_interpol), f_cubica(x_interpol)


x_new = np.linspace(min(ix), max(ix), 512)
n_interpola = 512
dt_interpola = x_new[1]-x_new[0]

cuadratica, cubica = interpola(incompletos,x_new)

FT_cuadratica = FT(cuadratica)/n_interpola
FT_cubica = FT(cubica)/n_interpola

freq_intepola = darFreq(n_interpola, dt_interpola)


f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(freqmia, abs(FT_sy))
axarr[0].set_title('signal')
axarr[1].plot(freq_intepola, abs(FT_cuadratica))
axarr[1].set_title('cuadratica')
axarr[1].set_ylabel('FT')
axarr[2].plot(freq_intepola, abs(FT_cubica))
axarr[2].set_title('cubica')
axarr[2].set_xlabel('freq(Hz)')
f.savefig("CorrealSergio_TF_interpola.pdf")

print "La TF de la senal original es la que tiene menos picos de frecuencias, luego le sigue la cubica y despues la cuadratica, con mas picos. La razon de lo anterior es que las interpolaciones son aproximaciones asi que introducen ruido. Ademas, la interpolacion cuadratica es menos exacta que la cubica, por lo que la primera introduce mas ruido que la segunda."

# Hacemos filtros

FT_cuadratica_f1 = FT_cuadratica.copy()
FT_cubica_f1 = FT_cubica.copy()
FT_cuadratica_f2 = FT_cuadratica.copy()
FT_cubica_f2 = FT_cubica.copy()
FT_sy_f2 = FT_sy.copy()

FT_cuadratica_f1[abs(freq_intepola) > 1000] = 0
FT_cubica_f1[abs(freq_intepola) > 1000] = 0

FT_cuadratica_f2[abs(freq_intepola) > 500] = 0
FT_cubica_f2[abs(freq_intepola) > 500] = 0

FT_sy_f2[abs(freqmia) > 500] = 0

clean_cuadratica_f1 = np.fft.ifft(FT_cuadratica_f1*n_interpola)
clean_cuadratica_f2 = np.fft.ifft(FT_cuadratica_f2*n_interpola)

clean_cubica_f1 = np.fft.ifft(FT_cubica_f1*n_interpola)
clean_cubica_f2 = np.fft.ifft(FT_cubica_f2*n_interpola)

clean_sy_f2 = np.fft.ifft(FT_sy_f2*nu)

f1, axarr1 = plt.subplots(2, sharex=True)
axarr1[0].plot(sx, np.real(clean_sy_f1), label = "signal")
axarr1[0].plot(x_new, np.real(clean_cuadratica_f1), label = "cuadratica")
axarr1[0].plot(x_new, np.real(clean_cubica_f1), label = "cubica")
axarr1[0].set_title('< 1000 Hz')

axarr1[1].plot(sx, np.real(clean_sy_f2), label = "signal")
axarr1[1].plot(x_new, np.real(clean_cuadratica_f2), label = "cuadratica")
axarr1[1].plot(x_new, np.real(clean_cubica_f2), label = "cubica")
axarr1[1].set_title('< 500 Hz')
axarr1[1].legend(loc="upper center",  bbox_to_anchor=(0.5, -0.05), ncol=3)
f1.savefig("CorrealSergio_2Filtros.pdf")