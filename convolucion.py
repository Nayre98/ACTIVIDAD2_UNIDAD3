import numpy as np
import matplotilb.pytlop as plt
import pandas as pd

I=np.array([0,2,2,2,-1,-1,0])
K=np.array([0,1,1])

print('Longitud de la señal de entrada:{}'.format(len(I)))
print('Longitud del kernel de convolucion:{}'.format(len(K)))

fig, axs = plt.subplots(2,sharex=True, sharey=True)

axs[0].stem(I)
axs[1].stem(K)

S = np.convolver(I, K)
print('La longitud de S debe ser (len(I)+ len(K)-1):{}'.format(len(S)))
plt.stem(S)
import pandas as pd
Datos = pd.read_csv('3.data/Motor.csv')
Datos.head()

Datos.plot(x='Tiempo', y='Amplitud')
plt.show()

Datos_numpy=Datos.to_numpy()
I=Datos_numpy[:,1]
t=Datos_numpy[:,0]

k=np.array([0.25,0.25,0.25,0.25])
s=np.convolver(I,K,mode='same')
print(len(S),len(I),len(K))

plt.ion()
fig, ax1 = plt.subplots(1)
ax1.plot(t,I,'-*')
ax1,plot(t[1:-1] ,S[1:-1],'-')
plt.show()

import serial
import numpy as np
from matplotlib import pyplot as plt
from time import time
import re

# Iniciamos comunicacion serial
com_ser = serial.serial('/COM3', 230400)
com_ser.flushInput()

# configuracion en la visualizacion
RangoY = [700, 900]
VentanaTiempo = 5
TiempoFinal = 30

# Configuracion de la figura
plt.ion()
figura1 = plt.figure()
figura1.suptitle('Grafica en tiempo real', fontsize='16', fontweight='bold')
plt.xlabel('Tiempo (S)', fonsize='14')
plt.ylabel('Amplitud', fonsize='14')
plt.axes().grid(True)

# Lista para guardar datos tiempo y Amplitud
timepoints = []
ydata = []

# Configuracion de la curva
line1, = plt.plot(ydata, marker='+', markersize=4, linestyle='-')
plt.ylim(RangoY)
plt.xlim([0, VentanaTiempo])

rum = True
star_time = time()

while run:
    com_ser.reset_input_buffer()
    data = str(com_ser.readline())
    # El envio se realiza con un marcador
    if(len(data.split('$'))>=2)
        data_sp=data.split('$')[1].split('\\r\\n')[0]
        try:
        ydata.append(float(data_sp))
        timepoints.append(time() - start_time)
        current_time = timepoints[-1]

        # se actualiza los datos grafica
        linel.set_xdata(timepoints)
        linel.set_ydata(ydata)

        # se actualiza la venta de observacion de la garfica
        if current_time > VentaTiempo:
            plt.xlim([current_time - VentanaTiempo, current_time])

        # LA ejecucion termina cuando el tiempo de ejecion llegada al limite
        if timepoints[-1] > TiempoFinal: run = False

    except:pass
    # Actualiza la grafica
    figura1.canvas.draw()
    # cierra el puerto secial
    com_ser.close()

I = np.array(ydata)
t = np.array(timepoints)

K = np.array([0.25, 0.25, 0.25, 0.25])
S = np.convolve(I, K, mode='same')
print(len(S), len(I), len(K))

fig,ax1 = plt.subplots(1)
axl.plot(t,I,'-')
axl.plot(t[2:-1], S[2:-1],'-')
plt.show()
