#this code is an example of propagation
import numpy as np
from vallado import vallado
import matplotlib.pyplot as plt
import math
# Declare some constants

def main():
    gravitation_constant = 398600
    # Initial values
    init_r = np.array([6671,0,0])
    init_v = np.array([0,7.72,0])

    init_r_2 = np.array([9371,0,0])
    init_v_2 = np.array([0,6.5219,0])

    init_r_transfer = init_r
    init_v_transfer = np.array([0,8.355,0])

    period = 2*math.pi*np.sqrt(6671**3/gravitation_constant)
    period_2 = 2*math.pi*np.sqrt(9371**3/gravitation_constant)
    transfer_time = 3574

    iterations = 10

    array_size = 100
    # Arrays for the time of flight
    tof_array = np.linspace(0, period, num=array_size)
    tof_array_2 = np.linspace(0, period_2, num=array_size)
    tof_transfer = np.linspace(0, transfer_time, num=array_size)

    # Iterate through and propagate for different times
    i = 0
    r = [[0]*3]*array_size
    v = [[0]*3]*array_size
    x = np.zeros(array_size)
    y = np.zeros(array_size)
    z = np.zeros(array_size)
    # Loop to get the initial orbit
    for temp_tof in tof_array:
        [r[i],v[i]] = vallado(k = gravitation_constant,r0 = init_r,v0 = init_v,tof = temp_tof, numiter = iterations)
        x[i] = r[i][0]
        y[i] = r[i][1]
        z[i] = r[i][2]
        i += 1

    i = 0
    x_2 = np.zeros(array_size)
    y_2 = np.zeros(array_size)
    z_2 = np.zeros(array_size)
    # Loop to get the target orbit
    for temp_tof in tof_array_2:
        [r[i],v[i]] = vallado(k = gravitation_constant,r0 = init_r_2,v0 = init_v_2,tof = temp_tof, numiter = iterations)
        x_2[i] = r[i][0]
        y_2[i] = r[i][1]
        z_2[i] = r[i][2]
        i += 1

    i = 0
    x_trans = np.zeros(array_size)
    y_trans = np.zeros(array_size)
    z_trans = np.zeros(array_size)
    # Loop to get the positions for the transfer
    for temp_tof in tof_transfer:
        [r[i],v[i]] = vallado(k = gravitation_constant,r0 = init_r_transfer,v0 = init_v_transfer,tof = temp_tof, numiter = iterations)
        x_trans[i] = r[i][0]
        y_trans[i] = r[i][1]
        z_trans[i] = r[i][2]
        i += 1

    # Once out of loop, plot it using matplot lib
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, label='Initial Orbit')
    ax.plot(x_2, y_2, z_2, label='Final Orbit')
    ax.plot(x_trans, y_trans, z_trans, label='Transfer Trajectory')
    ax.legend()
    # Add title and axes labels
    # Make the axes equal

    plt.show()


if __name__ == '__main__':
    main()
