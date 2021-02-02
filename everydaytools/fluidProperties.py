#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Copyright by Artur K. Lidtke, Univ. of Southampton, UK, 2016

This code is distributed under GNU Lesser General Public Licence Agreement 
version 3 or newer and without any warranty of merchantability or suitability
for any particular or general purpose.
Please see http://www.gnu.org/licenses/lgpl.html for details.

Created on Thu Jul 20 13:35:58 2017
"""

import numpy as np
import os
import pandas

def computeAirProperties(T, p, pInhPa=False):
    """ Calculates air density in kg/m3 and viscosity in m2/s given pressure
    in mmHg and temperature in deg C. Can also specify pressure in hPa with the flag."""
    mmHgTohPa = 1.3332239
    if pInhPa:
        p = p/mmHgTohPa
    
    # from engineering toolbox:
    # http://www.engineeringtoolbox.com/air-temperature-pressure-density-d_771.html
    rho = 1.325 * p/25.4 / ((T+273.15)*9/5) * 16.0185
    
    # from Sutherland equation:
    # http://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/fprops/propsoffluids/node5.html
    mu = 1.458e-6 * (T+273.15)**(3./2.) / ((T+273.15) + 110.4)
    
    # Alternative air density calculations as a function of temperature and pressure
    """
    # pressure is in mmHg, temperature in deg C
    # 1 mmHg = 133.322368 Pa
    # T_R = (T_C = 273.15) * 9/5
    
    # from Wikipedia: https://en.wikipedia.org/wiki/Density_of_air
    R = 287.058 # J / (kg K)
    rho0 = (p * 133.322368) / (R * (T+273.15))
    
    # from engineering toolbox: http://www.engineeringtoolbox.com/air-temperature-pressure-density-d_771.html
    rho1 = 1.325 * p/25.4 / ((T+273.15)*9/5) * 16.0185
    """
    return rho, mu/rho

#%% Read resource data.


# TODO need to fit polynomials to the ITTC data and be done with it? Or use interpolating splines in scipy

"""
# TODO this may fall over beacuse of relative paths?
dataDir = "../resources/fluidPropertyData"

files = os.listdir(dataDir)

data, units = {}, {}
for f in files:
    with open(os.path.join(dataDir, f), "r") as inStream:
        s = inStream.read().split("\n")
    fluid = f.split("_")[-1].replace(".csv","")
    labels = s[0].split(",")
    units[fluid] = dict(zip(labels, s[1].split(",")))
    values = np.array([[float(v) for v in l.split(",")] for l in s[2:] if l])
    data[fluid] = pandas.DataFrame(data=values, columns=labels)

#%% Unit test.
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()    
    nu = [computeAirProperties(t, 760.)[1] for t in data["air"]["T"]]
    plt.plot(data["air"]["T"], data["air"]["nu"], 'p')
    plt.plot(data["air"]["T"], nu)
    plt.xlabel("T {}".format(units["air"]["T"]))
    plt.ylabel("nu {}".format(units["air"]["nu"]))
    
    plt.figure()
    rho = [computeAirProperties(t, 760.)[0] for t in data["air"]["T"]]
    plt.plot(data["air"]["T"], data["air"]["rho"], 'p')
    plt.plot(data["air"]["T"], rho)
    plt.xlabel("T {}".format(units["air"]["T"]))
    plt.ylabel("rho {}".format(units["air"]["rho"]))

    plt.show()
"""