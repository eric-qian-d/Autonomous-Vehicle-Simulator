from pulp import *
import math
import random
import numpy as np


def simulate_rideshare(num_passengers, num_vehicles, vehicle_speed, x_max, y_max, time_length, time_interval, drop_off_pen, reassign_pen, wait_pen, pass1_pen = 1, pass2_pen = 1):
	
	'''
	Simulates a dynamic system for the specified length of time using the given inputs, updating information at every time interval. Returns
	the total amount of time waited
	'''

	def update_in_vehicle(R, R_IV, R_S, V_I, V_D): #need to consider order of dropoff for rideshares; FIX!
		if len(R_IV) == 0: return
		v_done, r_done = [], []
		for vehicle in V_D:
			passenger = R[vehicle.p]
			if dist_to_d(passenger, vehicle) < distance_travel:
				vehicle.x, vehicle.y = passenger.d[0], passenger.d[1]
				v_done.append(vehicle)
				r_done.append(passenger)
			else:
				x_comp = math.abs(vehicle.x - passenger.d[0])/(math.abs(vehicle.x - passenger.d[0]) + math.abs(vehicle.y - passenger.d[1]))
				y_comp = 1 - x_comp
				dx = np.sign(passenger.d[0] - vehicle.x) * x_comp * distance_travel
				dy = np.sign(passenger.d[1] - vehicle.y) * y_comp * distance_travel
				vehicle.x += dx
				vehicle.y += dy

		for vehicle in v_done:
			V_D.remove(vehicle)
			V_I.add(vehicle)
			vehicle.state = 'idle'

		for passenger in r_done:
			R_IV.remove(passenger)
			R_S.add(passenger)
			passenger.state = 'served'

	def update_assigned(R, R_A, R_IV, V_P, V_D):
		if len(R_A) == 0: return
		v_done, r_done = [], []
		for vehicle in V_P:
			passenger = R[vehicle.p]
			if distance(passenger, vehicle) < distance_travel:
				vehicle.x, vehicle.y = passenger.o[0], passenger.o[1]
				v_done.append(vehicle)
				r_done.append(passenger)
			else:
				x_comp = math.abs(vehicle.x - passenger.o[0])/(math.abs(vehicle.x - passenger.o[0]) + math.abs(vehicle.y - passenger.o[1]))
				y_comp = 1 - x_comp
				dx = np.sign(passenger.o[0] - vehicle.x) * x_comp * distance_travel
				dy = np.sign(passenger.o[1] - vehicle.y) * y_comp * distance_travel
				vehicle.x += dx
				vehicle.y += dy

		for vehicle in v_done:
			V_P.remove(vehicle)
			V_D.add(vehicle)
			vehicle.state = 'drop off'

		for passenger in r_done:
			R_A.remove(passenger)
			R_IV.add(passenger)
			passenger.state = 'in vehicle'

	def update_unassigned():
		if len(R_U) < 1: return
		solve_R_lessthan_V()

	def solve_R_lessthan_V(R, R_A, R_prime, V, V_P, V_prime):
		#Initialize variables
		d, y, d11, d12, d21, rideshare_pen = [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)]
		p, q = [0] * len(V), [0] * len(V)
		b, w = [0] * len(R), [0] * len(R)

		for j in range(num_vehicles):
			p[j] = 1 if V[j] in V_D else 0 
			q[j] = 1 if V[j] in V_P else 0

			for i in range(num_passengers):
				y[i][j] = 0
				if R[i].v == j:

					if V[j] in V_P:
						y[i][j] = 1

				d[i][j] = distance(R[i], V[j])

		for i in range(num_passengers):
			for j in range(num_vehicles):
				if V[j] in V_D and R[i] in R_prime:
					vehicle = V[j]
					current_passenger = R[V[j].p]
					considered_passenger = R[i]
					d11[i][j] = point_dist((vehicle.x, vehicle.y), current_passenger.d) + point_dist(considered_passenger.o, current_passenger.d) - dist_to_d(current_passenger, vehicle)
					d12[i][j] = point_dist(considered_passenger.o, current_passenger.d) + point_dist(current_passenger.d, considered_passenger.d) - point_dist(considered_passenger.o, considered_passenger.d)
					d21[i][j] = point_dist((vehicle.x, vehicle.y), current_passenger.d) + point_dist(considered_passenger.o, considered_passenger.d) + point_dist(considered_passenger.d, current_passenger.d) - dist_to_d(current_passenger, vehicle)
					rideshare_pen[i][j] = [psi1 * d11[i][j] + psi2 * d12[i][j], 0] if psi1 * d11[i][j] + psi2 * d12[i][j] < psi1 * d21[i][j] else [psi1 * d21[i][j], 1]
				else:
					d11[i][j] = 0
					d12[i][j] = 0
					d21[i][j] = 0
					rideshare_pen[i][j] = [0, None]

		for i in range(num_passengers):
			b[i] = R[i].reassigned
			w[i] = R[i].wait

		#Create new model for |R'| < |V'|
		model = pulp.LpProblem('R < V', LpMinimize)		

		#Set variables to optimize
		x = pulp.LpVariable.dicts('single_ride', ((i, j) for i in range(len(R)) for j in range(len(V))), lowBound = 0, upBound = 1, cat = 'Discrete')
		x_prime = pulp.LpVariable.dicts('shared_ride', ((i, j) for i in range(len(R)) for j in range(len(V))), lowBound = 0, upBound = 1, cat = 'Discrete')

		#Set objective function
		model += (pulp.lpSum([pulp.lpSum([[x[(i,j)] * (d[i][j] + phi * p[j])]
		+ [x_prime[(i,j)] * rideshare_pen[i][j][0]]
		+ [delta * q[j] * (x[(i,j)] + x_prime[(i,j)]) * (1 - y[i][j])]
		for i in range(len(R_prime))]) for j in range(len(V_prime))])), 'wait_cost'

		#Set constraints
		for j in range(num_vehicles):
			label = 'rideshare_pass_constraint_%d' % j
			condition = p[j] >= pulp.lpSum([[x_prime[(i,j)]] for i in range(len(R_prime))]) #passenger 1 before 2 ride-share
			model += condition, label

			label_single = 'one_max_single_%d' % j
			label_shared = 'one_max_shared_%d' % j
			condition_single = pulp.lpSum([[x[(i, j)]] for i in range(len(R_prime))]) <= 1 #cap the number of single rides for each vehicle
			condition_shared = pulp.lpSum([[x_prime[(i, j)]] for i in range(len(R_prime))]) <= 1 #cap the number of ride shares for each vehicle
			model += condition_single, label_single
			model += condition_shared, label_shared

		for i in range(num_passengers):
			label = 'all_assigned_%d' % i
			condition = pulp.lpSum([[x[(i,j)]] + [x_prime[(i,j)]] for j in range(len(V_prime))]) == 1 #every passenger is assigned
			model += condition, label

		for i in range(num_passengers):
			for j in range(num_vehicles):
				label_single = 'nonneg_single_' + str(i) + '_' + str(j)
				label_shared = 'nonneg_shared_' + str(i) + '_' + str(j)
				condition_single = x[(i,j)] >= 0 #nonnegative single rides
				condition_shared = x_prime[(i,j)] >= 0 #nonnegative ride shares
				model += condition_single, label_single
				model += condition_shared, label_shared

		for i in range(len(R_A)):
			label = 'stay_assigned_%d' % i
			condition = pulp.lpSum([x[(i,j)]] + [x_prime[(i,j)]]) #prevents assigned -> unassigned
			model += condition, label

			for j in range(len(V_P)):
				label_single = 'reassign_single_' + str(i) + '_' + str(j)
				label_shared = 'reassign_shared_' + str(i) + '_' + str(j)
				condition_single = b[i] * (y[i][j] - x[(i,j)]) <= 0  #prevents more than 1 reassignment
				condition_shared = b[i] * (y[i][j] - x_prime[(i,j)]) <= 0  #prevents more than 1 reassignment
				model += condition_single, label_single
				model += condition_shared, label_shared


		LpSolverDefault.msg = 1
		model.solve()
		for var in model.variables():
			if var.varValue != 0:
				if 'single_ride' in var.name:
					first_start = var.name.indexOf('(') + 1 #13
					first_end = var.name.indexOf(',') - 1
					second_start = var.name.indexOf(',') + 2
					second_end = var.name.indexOf(')') - 1
				else:


			print(var.name + ' = ' + str(var.varValue))

	#set constants
	phi = drop_off_pen
	delta = reassign_pen
	gamma = wait_pen
	psi1 = pass1_pen
	psi2 = pass2_pen
	T = time_length
	t = time_interval
	t_int = time_interval
	distance_travel = t_int * vehicle_speed

	#initiate passenger sets
	R, R_U, R_A, R_IV, R_S, R_prime = [], set(), set(), set(), set(), set()
	for passenger in range(num_passengers):
		p = Passenger(passenger, x_max, y_max, T)
		R.append(p)
		if p.appear >= time_interval:
			R_prime.add(p)
			R_U.add(p)

	#initiate vehicle sets
	V, V_I, V_P, V_D = [], set(), set(), set()
	V_prime = V #this is specific to this model; may change so I'm distinguishing V_prime and V
	for vehicle in range(num_vehicles):
		v = Vehicle(vehicle, x_max, y_max)
		V.append(v)

	#loop
	while t <= T:
		solve_R_lessthan_V(R, R_A, R_prime, V, V_P, V_prime)
		t += t_int
		for p in R:
			if p.appear >= t:
				R_prime.add(p)
				R_U.add(p)


class Passenger:
	def __init__(self, num, x_max, y_max, time_horizon):
		self.num = num
		self.x = random.random() * x_max
		self.y = random.random() * y_max
		self.o = (self.x, self.y)
		self.d = (random.random() * x_max, random.random() * y_max)
		self.state = 'unassigned'
		self.vehicle = None 
		self.reassigned = 0
		self.wait = 0
		self.appear = random.random() * time_horizon

	def v(self):
		return self.vehicle

class Vehicle:
	def __init__(self, num, x_max, y_max):
		self.num = num
		self.x = random.random() * x_max
		self.y = random.random() * y_max
		self.state = 'idle'
		self.passengers = []

	def p(self):
		return self.passengers

def distance(passenger, vehicle): #really doesn't matter which order arguments are given
	x_d = passenger.x - vehicle.x 
	y_d = passenger.y - vehicle.y
	return math.sqrt(x_d**2 + y_d**2)

def dist_to_d(passenger, vehicle): #order of arguments does matter
	x_d = passenger.d[0] - vehicle.x
	y_d = passenger.d[1] - vehicle.y
	return math.sqrt(x_d**2 + y_d**2)

def point_dist(p1, p2):
	x_d = p1[0] - p2[0]
	y_d = p1[1] - p2[1]
	return math.sqrt(x_d**2 + y_d**2)

#Where you can choose inputs - NOTE VEHICLES MUST BE GREATER THAN PASSENGERS:
number_of_passengers = 5
number_of_vehicles = 5
vehicle_speed = 2
x_size = 40
y_size = 40
run_horizon = 1
update_interval = 1
dropoff_reasssignment_penalty = 1
reassignment_penalty = 1
waiting_penalty = 1
pass1_distance_pen = 1
pass2_distance_pen = 1

simulate_rideshare(number_of_passengers, number_of_vehicles, vehicle_speed, x_size, y_size, run_horizon, update_interval, dropoff_reasssignment_penalty, reassignment_penalty, waiting_penalty, pass1_distance_pen, pass2_distance_pen)