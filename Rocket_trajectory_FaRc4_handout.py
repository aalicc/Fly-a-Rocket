# -*- coding: utf-8 -*-
"""
Numerical calculation of rocket trajectory with air resistance.
In 2 dimensions, and with a time-dependent mass.

Created on Thu 07 Dec 2023 at 14:50:30.
Last modified [dd.mm.yyyy]: 15.12.2023
@author: bjarne.aadnanes.bergtun
"""

import numpy as np # maths
import matplotlib.pyplot as plt # plotting


# ========================= Constants & parameters ========================== #

# Constants
g = 9.81				# [m/s^2]
rho_0 = 1.225			# [kg/m^3]
H =	7700       	        # [m]
C_D = 0.51				# [-]
A = 1.081e-2          	# [m^2]
m_0 = 19.765			# [kg]
m_f = 11.269			# [kg]
T_0 = 2.5018e+3		    # [N]
t_b = 6.09		    	# [s]
theta_0 = 75*np.pi/180  # [rad]


# Simulation parameters
dt = 0.001				# simulation time step [s]
#t_0 = 0                 # simulation start [s]; not needed when we start at 0
t_f = 180				# simulation time end [s]

# ================================ Functions ================================ #

def m(t):
    """
    Rocket mass [kg]
    as a function of time t [s]
    PS! Assumes 0 <= t <= t_b.
    """
    return m_0 - (m_0 - m_f) * t / t_b


def rho(y):
    """
    Air density [kg/m^3]
    as a function of altitude y [m]
    """
    return rho_0 * np.exp(-y/H)


def D_y(t, y, v, v_y):
    """
    Acceleration in the y-direction due to air resistance [m/s^2]
    as a function of time [s], altitude y [m], and velocity v, v_y [m/s]
    """
    return -0.5 * C_D * A * rho(y) * v * v_y

def D_x(t, y, v, v_x):
    """
    Acceleration in the y-direction due to air resistance [m/s^2]
    as a function of time [s], altitude y [m], and velocity v, v_y [m/s]
    """
    return -0.5 * C_D * A * rho(y) * v * v_x

# ======================== Numerical implementation ========================= #

# Calculate the number of data points in our simulation
N = int(np.ceil(t_f/dt))
T_y = T_0*np.sin(theta_0)
T_x = T_0*np.cos(theta_0)

# Create data lists
# Except for the time list, all lists are initialized as lists of zeros.
t = np.arange(t_f, step=dt) # runs from 0 to t_f with step length dt
y = np.zeros(N)
v_y = np.zeros(N)
a_y = np.zeros(N)

x = np.zeros(N)
v_x = np.zeros(N)
a_x = np.zeros(N)

# We will use while loops to iterate over our data lists. For this, we will use
# the auxillary variable n to keep track of which element we're looking at.
# The data points are numbered from 0 to N-1
n = 0
n_max = N - 1


# Thrusting phase
# ---------------------------------- #
# First, we iterate until the motor has finished burning, or until we reach the lists' end:
while t[n] < t_b and n < n_max:
    # Values needed for Euler's method
    # ---------------------------------- #

    # Speed
    v = np.sqrt((v_x[n]**2)+(v_y[n]**2)) # Powers, like a^2, is written a**2

    # Acceleration
    a_y[n] = ( T_y + D_y(t[n], y[n], v, v_y[n]) )/ m(t[n]) - g
    a_x[n] = ( T_x + D_x(t[n], y[n], v, v_x[n]) )/ m(t[n])
    
    # Euler's method:
    # ---------------------------------- #
    # Position
    y[n+1] = y[n] + v_y[n]*dt
    x[n+1] = x[n] + v_x[n]*dt
    
    # Velocity
    v_y[n+1] = v_y[n] + a_y[n]*dt
    v_x[n+1] = v_x[n] + a_x[n]*dt
    
    #theta_0 = np.arctan2(v_y[n+1], v_x[n+1])

    # Advance n with 1
    n += 1


# Coasting phase
# ---------------------------------- #
# Then we iterate until the rocket has crashed, or until we reach the lists' end:
while y[n] >= 0 and n < n_max:
    # Values needed for Euler's method
    # ---------------------------------- #
    # Speed
    v = np.sqrt((v_x[n]**2)+(v_y[n]**2))

    # Acceleration
    a_y[n] = D_y(t[n], y[n], v, v_y[n]) / m_f - g
    a_x[n] = D_x(t[n], y[n], v, v_x[n]) / m_f
    
    # Euler's method:
    # ---------------------------------- #
    # Position
    y[n+1] = y[n] + v_y[n]*dt
    x[n+1] = x[n] + v_x[n]*dt
    # Velocity
    v_y[n+1] = v_y[n] + a_y[n]*dt
    v_x[n+1] = v_x[n] + a_x[n]*dt
        
    # Advance n with 1
    n += 1
 
# When we exit the loops above, our index n has reached a value where the rocket
# has crashed (or it has reached its maximum value). Since we don't need the
# data after n, we redefine our lists to include only the points from 0 to n:
t = t[:n]
y = y[:n]
v_y = v_y[:n]
a_y = a_y[:n]
x = x[:n]
v_x = v_x[:n]
a_x = a_x[:n]


# ============================== Data analysis ============================== #
# Apogee
n_a = np.argmax(y) # Index at apogee


# =========================== Printing of results =========================== #

print('\n---------------------------------\n')
print('Apogee time:\t', t[n_a], 's')
print('... altitude:\t', round(y[n_a])/1000, 'km')
print('\n---------------------------------\n')


# =========================== Plotting of results =========================== #
# Close all currently open figures , so we avoid mixing up old and new figures.
plt.close('all')

'''
plt.figure('Trajectory')
plt.plot (t, y)    
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
'''

plt.figure('Speed')
v = np.sqrt((v_x**2)+(v_y**2))
plt.plot (t, v) # Speed graph
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")


'''
plt.figure('Acceleration')
plt.plot (t, (acc_x/g))
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s**2]")
'''

plt.grid(linestyle='--')
plt.show()