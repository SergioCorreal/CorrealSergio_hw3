from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt
from scipy import fftpack
from matplotlib import cm
from matplotlib.colors import LogNorm

img = plt.imread("Arboles.png")
FT_img = fftpack.fft2(img, axes=(0, 1))

N = img.shape[0]
freqx = np.fft.fftfreq(N,1)

plt.figure()
for i in range(N):
	plt.plot(freqx,abs(FT_img[i,:]))
plt.xlabel("frecuencia supuesta")
plt.ylabel("FT")
plt.savefig("CorrealSergio_FT2D.pdf")
plt.close()

print "Todas las senales altas corresponden a colores osilatorios en la grafica que se desean remover. Por eso, remuevo todas estas senales simplemente haciendo un corte de intensidades."

for i in range(N):
    for j in range(N):
        if(abs(FT_img[i][j])>800):
            FT_img[i][j]=0

plt.figure()
plt.imshow(np.abs(FT_img), norm=LogNorm(vmin=5))
plt.title("TF")
plt.colorbar()
plt.savefig("CorrealSergio_FT2D_filtrada.pdf")
plt.close()

img_f = fftpack.ifft2(FT_img).real

plt.figure()
plt.imshow(img_f, plt.cm.gray)
plt.title('Filtered Image')
plt.savefig("CorrealSergio_Imagen_filtrada.pdf")
plt.close()


"""
esto es otra manera en que intente filtrar
m = 0.4
clean_FT_img = FT_img.copy()
f = FT_img.shape[0]
c = FT_img.shape[1]

clean_FT_img[int(f*m):int(f*(1-m))] = 0
clean_FT_img[:, int(c*m):int(c*(1-m))] = 0

plt.figure()
plt.imshow(np.abs(FT_img), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Clean FT')
plt.show()
plt.close()
"""