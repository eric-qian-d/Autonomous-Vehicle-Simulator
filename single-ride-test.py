from pulp import *
import math
import random
import numpy as np


def simulate_rideshare(num_passengers, num_vehicles, vehicle_speed, x_max, y_max, time_length, time_interval, drop_off_pen, reassign_pen, wait_pen, pass1_pen = 1, pass2_pen = 1):
	
	'''
	Simulates a dynamic system for the specified length of time using the given inputs, updating information at every time interval. Returns
	the total amount of time waited
	'''

	#need to still account for the d variable for passenger 1's
	#need to adjust first two updates to account for ride shares and if the driver is diverting for a pickup

	def update_in_vehicle(R, R_IV, R_S, V_I, V_D): #need to consider order of dropoff for rideshares; FIX!
		if len(R_IV) == 0: return
		print('In the in vehicle update function')
		v_done, r_done = [], []
		for vehicle in V_D:
			passenger = R[vehicle.passengers[vehicle.serving]]
			if dist_to_d(passenger, vehicle) < distance_travel:
				vehicle.x, vehicle.y = passenger.d[0], passenger.d[1]
				v_done.append(vehicle)
				r_done.append(passenger)
				print(vehicle, passenger)
			else:
				x_comp = abs(vehicle.x - passenger.d[0])/(abs(vehicle.x - passenger.d[0]) + abs(vehicle.y - passenger.d[1]))
				y_comp = 1 - x_comp
				dx = np.sign(passenger.d[0] - vehicle.x) * x_comp * distance_travel
				dy = np.sign(passenger.d[1] - vehicle.y) * y_comp * distance_travel
				vehicle.x += dx
				vehicle.y += dy
				for p in vehicle.passengers:
					R[p].x += dx
					R[p].y += dy

		for vehicle in v_done:
			V_D.remove(vehicle)
			V_I.add(vehicle)
			vehicle.passengers.pop(vehicle.serving)
			if len(vehicle.passengers) > 0:
				vehicle.serving = 0
			else:
				vehicle.serving = None
				vehicle.state = 'idle'

		for passenger in r_done:
			R_IV.remove(passenger)
			R_S.add(passenger)
			passenger.state = 'served'

	def update_assigned(R, R_A, R_IV, V_P, V_D):
		if len(R_A) == 0: return
		print('In the assigned update function')
		v_done, r_done = [], []
		for vehicle in V_P:
			passenger = R[vehicle.passengers[vehicle.picking_up]]
			if distance(passenger, vehicle) < distance_travel:
				vehicle.x, vehicle.y = passenger.o[0], passenger.o[1]
				v_done.append(vehicle)
				r_done.append(passenger)
			else:
				x_comp = abs(vehicle.x - passenger.o[0])/(abs(vehicle.x - passenger.o[0]) + abs(vehicle.y - passenger.o[1]))
				y_comp = 1 - x_comp
				dx = np.sign(passenger.o[0] - vehicle.x) * x_comp * distance_travel
				dy = np.sign(passenger.o[1] - vehicle.y) * y_comp * distance_travel
				vehicle.x += dx
				vehicle.y += dy
				for p in vehicle.passengers:
					if R[p] in R_IV:
						R[p].x += dx
						R[p].y += dy

		for vehicle in v_done:
			V_P.remove(vehicle)
			V_D.add(vehicle)
			vehicle.state = 'drop off'
			vehicle.picking_up = None

		for passenger in r_done:

			R_A.remove(passenger)
			R_IV.add(passenger)
			R_prime.remove(passenger)
			passenger.state = 'in vehicle'

	# def update_unassigned():
	# 	if len(R_U) < 1: return
	# 	solve_R_lessthan_V()

	def solve_R_lessthan_V(R, R_A, R_prime, V, V_P, V_prime, t):
		if len(R_prime) < 1: return
		print('passenger numbers')
		for passenger in R_prime:
			print(passenger.num)

		#Initialize variables
		d, y = [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)] 
		d11, d12, d21, rideshare_pen = [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)]
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
					current_passenger = R[V[j].passengers[0]]
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




		#Verifying that things were initialized properly

		# print('checking p')
		# for j in range(num_vehicles):
		# 	print(p[j])

		# print('checking q')
		# for j in range(num_vehicles):
		# 	print(q[j])

		# print('checking d')
		# for i in range(num_passengers):
		# 	for j in range(num_vehicles):
		# 		print(d[i][j])


		#Create new model for |R'| < |V'|
		model = ''
		model = pulp.LpProblem('R_lessthan_V_' + str(t), LpMinimize)		

		#Set variables to optimize
		x = pulp.LpVariable.dicts('single_ride_' + str(t), ((i.num, j.num) for i in R_prime for j in V_prime), cat = 'Binary')
		x_prime = pulp.LpVariable.dicts('shared_ride_' + str(t), ((i.num, j.num) for i in R_prime for j in V_prime), cat = 'Binary')

		#Set objective function
		model += (pulp.lpSum([pulp.lpSum([[x[(i.num, j.num)] * (d[i.num][j.num] + phi * p[j.num])]
		+ [x_prime[(i.num, j.num)] * rideshare_pen[i.num][j.num][0]]
		+ [delta * q[j.num] * (x[(i.num, j.num)] + x_prime[(i.num, j.num)]) * (1 - y[i.num][j.num])]
		for i in R_prime]) for j in V_prime])), 'wait_cost_' + str(t)

		#Set constraints
		for j in V_prime:
			label = 'rideshare_pass_constraint_time%d_v%d' % (t, j.num)
			condition = p[j.num] >= pulp.lpSum([[x_prime[(i.num, j.num)]] for i in R_prime]) #passenger 1 before 2 ride-share
			model += condition, label

			label_single = 'one_max_single_time%d_v%d' % (t, j.num)
			label_shared = 'one_max_shared_time%d_v%d' % (t, j.num)
			condition_single = pulp.lpSum([[x[(i.num, j.num)]] for i in R_prime]) <= 1 #cap the number of single rides for each vehicle
			condition_shared = pulp.lpSum([[x_prime[(i.num, j.num)]] for i in R_prime]) <= 1 #cap the number of ride shares for each vehicle
			model += condition_single, label_single
			model += condition_shared, label_shared

		for i in R_prime:
			label = 'all_assigned_time%d_p%d' % (t, i.num)
			condition = pulp.lpSum([[x[(i.num, j.num)]] + [x_prime[(i.num, j.num)]] for j in V_prime]) == 1 #every passenger is assigned
			model += condition, label

		for i in R_prime:
			for j in V_prime:
				label_single = 'nonneg_single_time%d_p%d_i%d' % (t, i.num, j.num)
				label_shared = 'nonneg_shared_time%d_p%d_i%d' % (t, i.num, j.num)
				condition_single = x[(i.num, j.num)] >= 0 #nonnegative single rides
				condition_shared = x_prime[(i.num, j.num)] >= 0 #nonnegative ride shares
				model += condition_single, label_single
				model += condition_shared, label_shared

		for i in R_A:
			label = 'stay_assigned_time%d_p%d' % (t, i.num)
			condition = pulp.lpSum([x[(i.num, j.num)]] + [x_prime[(i.num, j.num)]]) #prevents assigned -> unassigned
			model += condition, label

			for j in V_P:
				label_single = 'reassign_single_time%d_p%d_i%d' % (t, i.num, j.num)
				label_shared = 'reassign_shared_time%d_p%d_i%d' % (t, i.num, j.num)
				condition_single = b[i.num] * (y[i.num][j.num] - x[(i.num,j.num)]) <= 0  #prevents more than 1 reassignment
				condition_shared = b[i.num] * (y[i.num][j.num] - x_prime[(i.num,j.num)]) <= 0  #prevents more than 1 reassignment
				model += condition_single, label_single
				model += condition_shared, label_shared


		# LpSolverDefault.msg = 1
		model.solve()
		
		for var in model.variables():
			first_start = var.name.find('(') + 1 #13
			first_end = var.name.find(',') - 1
			second_start = var.name.find(',') + 2
			second_end = var.name.find(')') - 1 #len(var.name) - 1
			p = int(var.name[first_start: first_end + 1])
			v = int(var.name[second_start: second_end + 1])
			passenger = R[p]
			vehicle = V[v]
			if var.varValue == 1:
				if passenger in R_A:  #reconfigures vehicle and passenger info due to reassignments
					passenger.reassigned = 1
					if V[passenger.vehicle] in V_P:
						if len(V[passenger.vehicle].passengers) == 1: #reassigned passenger was reassigned from a single ride
							if p == V[passenger.vehicle].passengers[0]: #means we're not overwriting new information
								V_I.add(V[passenger.vehicle])
								V_P.remove(V[passenger.vehicle])
								V[passenger.vehicle].picking_up = None
								V[passenger.vehicle].state = 'idle'
								V[passenger.vehicle].passengers = []
								V[passenger.vehicle].serving = None
						else: #reassigned passenger was reassigned from the second passenger of a shared ride
							if p == V[passenger.vehicle].passengers[0]: #means we're not overwriting new information
								V_D.add(V[passenger.vehicle])
								V_P.remove(V[passenger.vehicle])
								V[passenger.vehicle].picking_up = None
								V[passenger.vehicle].state = 'drop off'
								V[passenger.vehicle].passengers.pop(1)
								V[passenger.vehicle].serving = 0
					else: #reassigned passenger must've come from a vehicle in V_D
						if p == V[passenger.vehicle].next:
							V[passenger.vehicle].next = None


				else: #passenger was previously unassigned
					R_A.add(passenger)
					R_U.remove(passenger)
					passenger.state = 'assigned'

				if 'single_ride' in var.name: #case where a single ride is assigned/reassigned
					if vehicle in V_I: 
						V_I.remove(vehicle)
						V_P.add(vehicle)

					if vehicle in V_I or vehicle in V_P:	#passenger will be picked up immediately next
						vehicle.passengers = [p]
						vehicle.picking_up = 0
						vehicle.serving = 0
						vehicle.state = 'picking_up'
													
					else: #passenger will be picked up after the next passenger is dropped off
						vehicle.next = p
					
				else:
					if vehicle in V_P: #when a rideshare gets reassigned to another rideshare
						vehicle.passengers[1] = p

					else: #when a new rideshare assigned
						vehicle.passengers.append(p)

					vehicle.picking_up = 1
					vehicle.serving = 0 if rideshare_pen[p][v][1] == 0 else 1

				passenger.vehicle = v

			print(var.name + ' = ' + str(var.varValue))
		print("Total Cost = " + str(value(model.objective)))

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
	for p in range(num_passengers):
		passenger = Passenger(p, x_max, y_max, T)
		R.append(passenger)
		if passenger.appear <= time_interval:
			R_prime.add(passenger)
			R_U.add(passenger)

	#initiate vehicle sets
	V, V_I, V_P, V_D = [], set(), set(), set()
	V_prime = V #this is specific to this model; may change so I'm distinguishing V_prime and V
	for v in range(num_vehicles):
		vehicle = Vehicle(v, x_max, y_max)
		V.append(vehicle)
		V_I.add(vehicle)

	print()
	print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<') #for visibility
	print()

	#loop
	while t <= T:
		print()
		print('Modeling time = ' + str(t))
		print()
		for vehicle in V:
			print('Vehicle #' + str(vehicle.num) + ' pos: (' + str(vehicle.x) + ',' + str(vehicle.y) + ')')
		for passenger in R:
			print('Passenger #' + str(passenger.num) + ' pos: (' + str(passenger.x) + ',' + str(passenger.y) + ')')
		print('R_U' , R_U)
		print('R_A' , R_A)
		print('R_prime', R_prime)

		print()

		print('R_IV', R_IV)
		print('R_S' , R_S)

		print()

		print('V_I' , V_I)
		print('V_P' , V_P)
		print('V_D' , V_D)

		
		# print('Upadting in vehcile')
		update_in_vehicle(R, R_IV, R_S, V_I, V_D)
		# print('Updating assigned')
		update_assigned(R, R_A, R_IV, V_P, V_D)
		# print('Solving new optimization')
		solve_R_lessthan_V(R, R_A, R_prime, V, V_P, V_prime, t)
		t += t_int
		for passenger in R:
			if passenger.appear <= t and passenger.appear > t - t_int:
				R_prime.add(passenger)
				R_U.add(passenger)

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
		self.appear = random.random() * time_horizon * 0.2

	def v(self):
		return self.vehicle #vehicle number that is serving it; not the vehicle object!

class Vehicle:
	def __init__(self, num, x_max, y_max):
		self.num = num
		self.x = random.random() * x_max
		self.y = random.random() * y_max
		self.state = 'idle'
		self.passengers = [] #passenger number being served; not the passenger objects!
		self.picking_up = None #index in self.passengers that is being picked up
		self.serving = None #index in self.passengers that is getting dropped off next
		self.next = None #passenger number that will be picked up after current passengers are all dropped off

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
number_of_passengers = 2
number_of_vehicles = 2
vehicle_speed = 5
x_size = 40
y_size = 40
run_horizon = 6
update_interval = 1
dropoff_reasssignment_penalty = 1
reassignment_penalty = 1
waiting_penalty = 1
pass1_distance_pen = 1
pass2_distance_pen = 1

simulate_rideshare(number_of_passengers, number_of_vehicles, vehicle_speed, x_size, y_size, run_horizon, update_interval, dropoff_reasssignment_penalty, reassignment_penalty, waiting_penalty, pass1_distance_pen, pass2_distance_pen)