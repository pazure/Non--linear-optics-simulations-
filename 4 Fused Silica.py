
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
t0 = 6
dt = 2*np.pi/(OM)
L = 1e6 # Lenght of cristal


print('Time resolution dt ={0:0.3f} '.format(dt))
N = 2**17
dom = OM/N
om_c = 2*np.pi*c/wl
T = np.pi*2/dom
om = np.arange(-OM/2,OM/2,dom )


print('Time window T ={0:0.1f} '.format(T))
print('Carrier angular frequency w_c = {0:0.3f} '.format(om_c))
print('Frequency resolution {0:0.6f} '.format(dom))

AMP= (np.sqrt(np.pi)*t0)/(np.sqrt(8*np.log(2)))
num = (om-om_c)**2*t0**2 # Numerator of exp function
den =8*np.log(2)# Denominator of exp function


#input Electric field in spectrall domain

Ew = AMP* np.exp(-num/den)



i_min= np.argmin(np.abs(om-1))
i_max = np.argmin(np.abs(om-4))
i_om0 = np.argmin(np.abs(om-om_c))








def refindex(angular_freq, glass_type):
    if angular_freq == 0:
        return 1 
    else:
        x = (2 * np.pi * c / angular_freq) * 1e-3  # convert to micrometers
    if glass_type == 1 and x >= 0.21 and x <= 6.7:
        return (1 + 0.6961663 / (1 - (0.0684043 / x) ** 2) +
                0.4079426 / (1 - (0.1162414 / x) ** 2) +
                0.8974794 / (1 - (9.896161 / x) ** 2)) ** 0.5
    elif glass_type == 2 and x >= 0.38 and x <= 2.5:
        return (1 + 1.03961212 / (1 - 0.00600069867 / x ** 2) +
                0.231792344 / (1 - 0.0200179144 / x ** 2) +
                1.01046945 / (1 - 103.560653 / x ** 2)) ** 0.5
    elif glass_type == 3 and x >= 0.3 and x <= 2.5:
        return (1 + 1.62153902 / (1 - 0.0122241457 / x ** 2) +
                0.256287842 / (1 - 0.0595736775 / x ** 2) +
                1.64447552 / (1 - 147.468793 / x ** 2)) ** 0.5
    else:
        return 1 
    
    
    
def FWHM_one(I_t_n,time):
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
    t_numeric =  time[right_idx]-time[left_idx]
    
    return t_numeric


# Vectorize refractive index 

Fused = np.array([refindex(x,1) for x in om])
BK7 = np.array([refindex(x,2) for x in om])
SF10 = np.array([refindex(x,3) for x in om])



#Defining parameteres for Optical transfer fuction 
# N(\omega)

fiom = (om/c)*Fused*L


Aom = np.ones(N)

Hom =Aom*np.exp(-1j*fiom)


Eom_out = Hom * Ew

fiom_out = np.unwrap(-np.angle(Eom_out))

m = round(fiom_out[i_om0]/(2* np.pi))
fiom_out = fiom_out-m*2*np.pi



#Ploting out Efield 

plt.figure()
plt.subplot(211)
plt.title('Spectral Phase and Amplitude of output pulse')
plt.plot(om[i_min:i_max], np.abs(Eom_out[i_min:i_max]),'r-')
plt.ylabel('Spectrall Amplitude(a.u)')
plt.grid()

plt.subplot(212)
plt.plot(om[i_min:i_max], fiom[i_min:i_max],'b-')
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('Spectrall Phase(rad) ')
plt.grid()
plt.tight_layout()


Iom_out = np.abs(Eom_out)**2
GD_out  = np.gradient(fiom, dom)
relGD_out = GD_out - GD_out[i_om0]


print('-------------------------------')
print('-------------------------------')
print('Spectral Width FWHM = {0:0.2f}'.format( FWHM_one(Iom_out, om)))

print('-------------------------------')
print('-------------------------------')

plt.figure()
plt.subplot(211)
plt.title('Spectral Intensity  and  Relative Group Delay of Output Pulse')
plt.plot(om[i_min:i_max], Iom_out[i_min:i_max]/max(Iom_out[i_min:i_max]),'r-')
plt.ylabel('Spectrall Intensity(a.u)')
plt.grid()
    
plt.subplot(212)
plt.plot(om[i_min:i_max], GD_out[i_min:i_max],'b-')
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('GD($\omega$) ')
plt.grid()
plt.tight_layout()


def FWHM_index(I_t_n, time):
    half_I_t_n = np.amax(I_t_n)/2
    idx = np.argmax(I_t_n)

    while I_t_n[idx] >half_I_t_n:
        idx = idx-1
    left_idx = idx


    idx = np.argmax(I_t_n)
    
    while I_t_n[idx] >half_I_t_n:
        idx = idx+1
    right_idx = idx

    hw_index =  np.array([right_idx,left_idx])
    
    return hw_index






i0=FWHM_index(Iom_out, om)[0]
i1 =FWHM_index(Iom_out, om)[1]



print('Printing Delta GD ')

Delta_GD= GD_out[i0] - GD_out[i1]

print( 'Delta GD = ', Delta_GD)

print('-------------------------------')
print('-------------------------------')




GDD  = np.gradient(GD_out , dom)
GDD0 = GDD[i_om0]
GD0 = GD_out[i_om0]

print('GD near om_c = ', GD_out[i_om0] )

print('-------------------------------')
print('-------------------------------')
# print(GD0)

t_est = t0*np.sqrt(1+((4*np.log(2)*GDD0)/(t0**2))**2)
print('Estimated output pulse duration ={0:0.6f} fs '.format((t_est)))

plt.figure()
plt.subplot(211)
plt.title('Spectral Intensity  and  GDD Output Pulse')
plt.plot(om[i_min:i_max], Iom_out[i_min:i_max]/max(Iom_out[i_min:i_max]),'r-')
plt.ylabel('Spectrall Intensity(a.u)')
plt.grid()
    
plt.subplot(212)
plt.plot(om[i_min:i_max], GDD[i_min:i_max],'b-')
plt.xlabel('Angular frequency(PHz)')
plt.ylabel('GD($\omega$) ')
plt.grid()
plt.tight_layout()



#-------------------------------------------------------
# Calculating temporal electric field                  -
#--------------------------------------------------------

time = np.arange(-T/2,T/2, dt)

t_min = GD0-3*t_est
t_max =  GD0+3*t_est
f0 = np.argmin(np.abs(time-t_min))
f1 = np.argmin(np.abs(time-t_max))
j_time0 = np.argmin(np.abs(time-0))




# Eom_out = Hom * Ew
# E_OOM = np.fft.fftshift(Eom_out)
# E_t_0 = OM/(np.pi*2)* np.fft.ifft(E_OOM)
# t_0 = np.fft.fftfreq(len(Eom_out), dom/(np.pi*2))
# E_t_out = np.fft.fftshift(E_t_0)


E_t_out =OM/(np.pi*2)*np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Eom_out)))




# n = round(phase[j_time0]/(2*np.pi))
# phase = phase-n*2*np.pi

# print((t_min, t_max))

#------------------------------------------
#------------------------------------------
# plotting electric field 
# -----------------------------------------
# -----------------------------------------

plt.figure()
plt.subplot(211)
plt.title('Electric field and temporal phase')
plt.plot(time[f0:f1], E_t_out[f0:f1],'r-')
plt.plot(time[f0:f1], np.abs(E_t_out[f0:f1]),'r--')
plt.xlabel('Time (fs)')
plt.ylabel('Electric field')
# plt.xlim(t_min, t_max)
plt.grid()

temporal_phase = np.unwrap(np.angle(E_t_out))

plt.subplot(212)
plt.plot(time[f0:f1], temporal_phase[f0:f1],'b-')
plt.xlabel('Time(fs)')
plt.ylabel('Temporal Phase $\Phi$ (rad)')
# plt.xlim(t_min, t_max)
plt.grid()
plt.tight_layout()







I_out = np.abs(E_t_out)**2 # Out put intensity

om_ins = np.gradient(temporal_phase, dt) # Instantaneous Angular frequency 


#defining a function for FWHM

print('-------------------------------')
print('-------------------------------')

print('Numerical output pulse duration = ', FWHM_one(I_out,time))

# del_om =4*np.log(2)/t0
# t_analytic  = del_om *GDD0
# print('Analytic Pulse duration = ',t_analytic )
# print('Numerical  Pulse duration = ', FWHM(I_out))

# print(' The difference between Numerical and Analytical  Pulse duration = ', abs(FWHM(I_out))-abs(t_analytic))




plt.figure()

plt.subplot(211)
plt.title('Temporal Intensity  and Instantaneous frequency ')

plt.plot(time[f0:f1], I_out[f0:f1],'r-')
plt.xlabel('Time(fs)')
plt.ylabel('Intensity(a.u)')
plt.grid()

plt.subplot(212)
plt.plot(time[f0:f1], om_ins[f0:f1],'b-')
plt.xlabel('Time(fs)')
plt.ylabel('Inst. ang. freq.(PHz)')
plt.grid()
plt.tight_layout()

print('-------------------------------')
print('-------------------------------')

print('Difference between estimated and FWHM output pulse duration ', t_est - FWHM_one(I_out,time))








