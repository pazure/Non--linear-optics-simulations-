# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:43 2024

@author: azure
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
c= 299.792458
wl =800
OM = 50 #PHZ
t0 = 10
dt = 2*np.pi/(OM)


print('Time resolution dt ={0:0.3f} '.format(dt))
N = 2**16
dom = OM/N
om_c = 2*np.pi*c/wl
T = np.pi*2/dom
om = np.arange(-OM/2,OM/2,dom )


print(' Time window T ={0:0.1f} '.format(T))
print('Carrier angular frequency w_c = {0:0.3f} '.format(om_c))
print('Frequency resolution {0:0.6f} '.format(dom))

AMP= (np.sqrt(np.pi)*t0)/(np.sqrt(8*np.log(2)))
num = (om-om_c)**2*t0**2 # Numerator of exp function
den =8*np.log(2)# Denominator of exp function


Ew = AMP* np.exp(-num/den)

plt.figure(1)
plt.title('Spectral Amplitude')
plt.plot(om, np.abs(Ew),'r-')
plt.xlim(1.8,3)
plt.ylim(0,8)
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('Spectrall Amplitude(a.u)')
plt.grid()

plt.figure(2)
plt.plot(om, np.angle(Ew),'b-')
plt.xlim(1.8,3)
plt.ylim(-1,1)
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('Spectrall Phase(rad) ')
plt.title('Spectral Phase')
plt.grid()


plt.figure(3)
plt.subplot(211)
plt.title('Spectral Phase and Amplitude')
plt.plot(om, np.abs(Ew),'r-')
plt.xlim(1.8,3)
plt.ylim(0,8)
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('Spectrall Amplitude(a.u)')
plt.grid()

plt.subplot(212)
plt.plot(om, np.angle(Ew),'b-')
plt.xlim(1.8,3)
plt.ylim(-1,1)
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('Spectrall Phase(rad) ')
plt.xlim(1.8,3)
plt.grid()
plt.tight_layout()


plt.figure(4)
plt.plot(om, np.abs(Ew)**2/max(np.abs(Ew)**2),'b-')
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('Spectrall Intensity(a.u) ')
plt.xlim(1.8,3)
plt.ylim(0,1)
plt.grid()


#Calculating E(t) from E(w)

# E_OOM = np.fft.fftshift(Ew)
# E_t_0 = OM/(np.pi*2)* np.fft.ifft(E_OOM)
# t_0 = np.fft.fftfreq(len(Ew), dom/(np.pi*2))
# E_t = np.fft.fftshift(E_t_0)
time = np.arange(-T/2,T/2, dt)
E_t =OM/(np.pi*2)*np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ew)))


E_ta = 0.5*np.exp(-2*np.log(2)*(time/t0)**2) *np.exp(1j*om_c*time)

plt.figure(5)
plt.plot(time, np.real(E_t),'b-', label = 'Numerical ')
plt.plot(time, np.real(E_ta),'r--', label = 'Analytical')
plt.xlim(-20,20)
plt.xlabel('Time(fs)')
plt.ylabel('Electric Field ')
plt.legend()
plt.grid()

left_lim = -20
right_lim = 20

i0 = np.argmin(np.abs(time-left_lim))
i1 = np.argmin(np.abs(time-right_lim))

plt.figure(8)
plt.subplot(211)
plt.title('Temporal Phase and Amplitude')
plt.plot(time[i0:i1], np.abs(E_t[i0:i1])**2,'r-')

plt.xlabel('Time (s)')
plt.ylabel('Temporal Amplitude')
plt.grid()

fit= np.unwrap(np.angle(E_t[i0:i1]))

plt.subplot(212)
plt.plot(time[i0:i1], np.unwrap(np.angle(E_t[i0:i1])),'b-')
plt.xlabel('Time(s)')
plt.ylabel('Temporal Phase(rad)  ')
plt.grid()
plt.tight_layout()



# Homework 2A

I_t = np.absolute(E_t)**2 # Intensity 
# Normalising the intensity
I_t_n = I_t/max(I_t)

plt.figure(6)
plt.plot(time[i0:i1], I_t_n[i0:i1],'r-', label = 'Normalised Intensity')
plt.xlabel('Time(fs)')
plt.ylabel('Amplitude(a.u)')
plt.legend()
plt.grid()




#Home work 2B

half_I_t_n = np.amax(I_t_n)/2
idx = np.argmax(I_t_n)

while I_t_n[idx] >half_I_t_n:
    idx = idx-1
left_idx = idx


idx = np.argmax(I_t_n)
while I_t_n[idx] >half_I_t_n:
    idx = idx+1
right_idx = idx

#
tau0_calculated =  time[right_idx]-time[left_idx]

print('t0 calculate = {0:0.4f} fs '.format(tau0_calculated))
print('Difference between t0 defined and  tau0 calculated = {0:0.4f} '.format(tau0_calculated-t0))


plt.figure(10)
plt.subplot(211)
plt.title('Temporal Phase and Amplitude')
plt.plot(time[i0:i1], np.abs(E_t[i0:i1])**2,'r-')

plt.xlabel('Time (s)')
plt.ylabel('Temporal Amplitude')
plt.grid()

fit= np.unwrap(np.angle(E_t))
p=np.gradient(fit, dt)

plt.subplot(212)
plt.plot(time[i0:i1],  p[i0:i1],'b-')
plt.xlabel('Time(s)')
plt.ylabel('Temporal Phase(rad)  ')
plt.grid()
plt.ylim(2.35,2.36)
plt.tight_layout()


gradient = (fit[i1]-fit[i0])/(time[i1]-time[i0])
print('central frequency = {0:0.5f}'.format(gradient))


