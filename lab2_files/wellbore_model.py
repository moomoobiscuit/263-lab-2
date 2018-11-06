
#########################################################################################
#
# Function library for the AUTOUGH2 wellbore model
#
# 	Functions:
#		transf0: transform parameter vector to a new base adapted for automated calibration
#		transf1: transform parameter vector to their original base
#		wellbore_model: runs the AUTOUGH2 model
#		wellbore_obj: returns the objective function associated to the wellbore model
#
#########################################################################################

# import modules
import numpy as np
from t2data import *
from t2listing import *
import csv

# parameter space
xlim = [90., 125.]	# initial pressure range
ylim = [0.05, .5]	# inital steam fraction range


def transf0(theta):
	"""transform parameter vector to a new base adapted for automated calibration

    Args:
		theta (list): parameter vector in original base
    """
	return [(theta[0]-xlim[0])/(xlim[1]-xlim[0]), (theta[1]-ylim[0])/(ylim[1]-ylim[0])]
	
def transf1(X):
	"""transform parameter vector to their original base

    Args:
		X (list): parameter vector in adapted base
    """
	return [X[0]*(xlim[1]-xlim[0])+xlim[0], X[1]*(ylim[1]-ylim[0])+ylim[0]]

def wellbore_model(theta):
	"""Returns the modelled pressure and enthalpy production history

    Args:
		theta (list): parameter vector
    """
	
	# set the four calibration parameters
	P0 = theta[0]*1.e5	# initial reservoir pressure (Pa)
	x0 = theta[1]		# initial steam mass fraction
	porosity = .082		# initial porosity
	permeability = 2.76e-15	# initial permeability (m2)
	
	# write values in the AUTOUGH2 dat file
	dat = t2data('SKG9D.DAT')
	dat.grid.rocktype['IGNIM'].porosity = porosity
	dat.grid.rocktype['IGNIM'].permeability = [permeability, permeability, permeability]
	dat.parameter['default_incons'] = [P0, x0]
	dat.write('SKG9D_1.DAT')

	# run AUTOUGH2
	dat.run(simulator='AUTOUGH2_5.exe', silent = True)

	# time vector indicated when result must be returned
	t_data = []
	with open('SKG9D_enth.csv', 'r') as csvfile:
		reader=csv.reader(csvfile, delimiter=',')
		for row in reader:								
			t_data.append(float(row[0]))

	# read LISTING file
	lst = t2listing('SKG9D_1.LISTING', skip_tables = ['connection'])
	[(th, h), (tp, p)] = lst.history([('g', ('  A 1', 'WEL 1'), 'Enthalpy'), ('e', '  A 1', 'Pressure')])
	th = np.array(th)*(1/24.)*(1/3600.)	# convert time from seconds to days
	p = np.array(p)*1.e-5				# convert pressure from Pa to bars
	h = np.array(h)*1.e-3				# convert enthalpy from J/kg to kJ/kg
	list_index = [(np.abs(th-t)).argmin() for t in t_data if t<=th[-1]]	# index list when result must be returned
	list_t = th[list_index]
	list_h = h[list_index]
	list_p = p[list_index]
	
	# Delete negative pressures
	while list_p[-1] < 0.:
		list_t = list_t[:-2]
		list_h = list_h[:-2]
		list_p = list_p[:-2]
	
	return list_t, list_h, list_p

def wellbore_obj(X, model=None):
	"""Returns the objective function associated to the wellbore model

    Args:
		X (list): parameter vector in a base adapted for automated calibration
    """
	
	# runs the AUTOUGH2 model
	list_t, list_h, list_p = wellbore_model(transf1(X))
	
	# load production data
	t_data = []						# initiate time vector
	h_data = []						# initiate enthalpy vector
	with open('SKG9D_enth.csv', 'r') as csvfile:	# open enthalpy production history csv file
		reader=csv.reader(csvfile, delimiter=',')	# open reader
		for row in reader:						# go through each row
			t_data.append(float(row[0]))			# save time
			h_data.append(float(row[1]))			# save enthalpy
	p_data = []						# initiate pressure vector
	with open('SKG9D_press.csv', 'r') as csvfile:	# open pressure production history csv file
		reader=csv.reader(csvfile, delimiter=',')	# open reader
		for row in reader:						# go through each row								
			p_data.append(float(row[1]))			# save pressure
	
	# calculate the objective function
	obj = 0.	# initiate
	w = .046	# enthalpy weight
	for i in range(len(list_t)):	# runs through time list
		obj += ((list_h[i]-h_data[i])*w)**2		# add weighted squared enthalpy difference
	for i in range(len(list_p)):	# runs through time list
		obj += (list_p[i]-p_data[i])**2			# add squared pressure difference
	
	# add penalty for models not running to the end
	n_diff = len(t_data)-len(list_t)	# number of time dots not reached by the model
	obj += n_diff*800.					# penalty
	
	# reduce objective function value for practical reasons
	obj = obj*1.e-5
	return obj