import cplex
import math
import random
import numpy as np


def simulate_rideshare(num_passengers, num_vehicles, vehicle_speed, x_max, y_max, time_length, time_interval, drop_off_pen, reassign_pen, wait_pen, pass1_pen = 1):
	
	'''
	Simulates a dynamic system for the specified length of time using the given inputs, updating information at every time interval. Returns
	the total amount of time waited
	'''


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
			else:
				x_comp = abs(vehicle.x - passenger.d[0])/(abs(vehicle.x - passenger.d[0]) + abs(vehicle.y - passenger.d[1]))
				y_comp = 1 - x_comp
				theta = math.atan(y_comp/x_comp)
				dx = np.sign(passenger.d[0] - vehicle.x) * math.cos(theta) * distance_travel
				dy = np.sign(passenger.d[1] - vehicle.y) * math.sin(theta) * distance_travel
				vehicle.x += dx
				vehicle.y += dy
			for p in vehicle.passengers:
				if R[p] in R_IV:
					R[p].x, R[p].y = vehicle.x, vehicle.y

		for vehicle in v_done:
			V_D.remove(vehicle)
			vehicle.passengers.pop(vehicle.serving)
			
			if vehicle.next is not None:
				vehicle.serving = 0
				vehicle.passengers = [vehicle.next]
				vehicle.picking_up = 0
				vehicle.next = None
				V_P.add(vehicle)
				vehicle.state = 'drop off'
			else:
				V_I.add(vehicle)
				vehicle.serving = None
				vehicle.state = 'idle'

		for passenger in r_done:
			R_IV.remove(passenger)
			R_S.add(passenger)
			passenger.state = 'served'
			passenger.vehicle = None

	def update_assigned(R, R_A, R_IV, V_P, V_D):
		if len(R_A) == 0: return
		print('In the assigned update function')
		v_done, r_done = [], []
		for vehicle in V_P:
			print(vehicle.num, vehicle.passengers, vehicle.next, vehicle.serving, vehicle.picking_up, R[vehicle.passengers[0]].vehicle)
			passenger = R[vehicle.passengers[vehicle.picking_up]]
			if distance(passenger, vehicle) < distance_travel:
				vehicle.x, vehicle.y = passenger.o[0], passenger.o[1]
				v_done.append(vehicle)
				r_done.append(passenger)
			else:
				x_comp = abs(vehicle.x - passenger.o[0])/(abs(vehicle.x - passenger.o[0]) + abs(vehicle.y - passenger.o[1]))
				y_comp = 1 - x_comp
				theta = math.atan(y_comp/x_comp)
				dx = np.sign(passenger.o[0] - vehicle.x) * math.cos(theta) * distance_travel
				dy = np.sign(passenger.o[1] - vehicle.y) * math.sin(theta) * distance_travel
				vehicle.x += dx
				vehicle.y += dy

		for vehicle in v_done:
			V_P.remove(vehicle)
			V_D.add(vehicle)
			vehicle.state = 'drop off'
			vehicle.picking_up = None

		for passenger in r_done:
			print('removing passenger ', passenger, '   #', passenger.num )
			R_A.remove(passenger)
			R_IV.add(passenger)
			R_prime.remove(passenger)
			passenger.state = 'in vehicle'

	# def update_unassigned(R, R_A, R_IV, R_prime, V, V_P, V_prime, t):
	# 	if len(R_U) < 1: return
		# solve_R_greaterthan_V(R, R_A, R_IV, R_prime, V, V_P, V_prime, t) if len(R_prime) >= len(V_prime) else solve_R_lessthan_V(R, R_A, R_IV, R_prime, V, V_P, V_prime, t)


	def solve_R_lessthan_V(R, R_A, R_IV, R_prime, V, V_P, V_prime, t):
		if len(R_prime) < 1: return
		R_joined = R_prime.union(R_IV)

		#Initialize variables
		d, y = [[0 for j in range(num_vehicles)] for i in range(num_passengers)], [[0 for j in range(num_vehicles)] for i in range(num_passengers)] 
		p, q = [0] * len(V), [0] * len(V)
		b = [0] * len(R)

		for j in range(num_vehicles):
			p[j] = 1 if V[j] in V_D else 0 
			q[j] = 1 if V[j] in V_P else 0

			for i in range(num_passengers):
				y[i][j] = 0
				if R[i].vehicle == j:
					y[i][j] = 1
					print(i, j)

				if V[j] in V_D:
					d[i][j] = dist_to_d(R[V[j].passengers[V[j].serving]], V[j]) + point_dist(R[V[j].passengers[V[j].serving]].d, R[i].o)
				else:
					d[i][j] = distance(R[i], V[j])

		for i in range(num_passengers):
			b[i] = R[i].reassigned

		#Verifying that things were initialized properly

		# print('checking b')
		# for j in range(num_passengers):
		# 	print(b[j])

		# print('checking y')
		# for i in range(num_passengers):
		# 	for j in range(num_vehicles):
		# 		print('i: ', i, '  j: ', j, '  = ', y[i][j])

		# print('checking q')
		# for j in range(num_vehicles):
		# 	print(q[j])

		print('checking d')
		for i in range(num_passengers):
			for j in range(num_vehicles):
				print('pass ', i, 'veh ', j, ': ', d[i][j])

		#Create new model for |R'| < |V'|
		problem = cplex.Cplex()
    		problem.objective.set_sense(problem.objective.sense.minimize)

		#Set variables to optimize
		obj = []
		lb = []
		ub = []
		names = []
		variable_types = []

		for i in R_joined:
			for j in V_prime:
				names.append('x({0},{1})'.format(i.num, j.num))
				obj.append(d[i.num][j.num] + phi * p[j.num] + delta * q[j.num] * (1 - y[i.num][j.num]))
				lb.append(0)
				ub.append(1)
				variable_types.append("C")

		#Set constraints
		constraint_names = []
		constraints = []
		constraint_rhs = []
		constraint_sense = []

		for j in V_prime:
			#caps single rides at 1
			single_ride_cap_constraint = [[], []]
			for i in R_joined:
				single_ride_cap_constraint[0].append('x({0},{1})'.format(i.num, j.num))
				single_ride_cap_constraint[1].append(1)
				
			constraint_names.append('single_ride_cap_constraint_{0}'.format(j.num))
			constraints.append(single_ride_cap_constraint)
			constraint_rhs.append(1)
			constraint_sense.append('L')
			
		#makes sure that every passenger is assigned to exactly 1 vehicle since |R| < |V|
		for i in R_joined:
			passenger_assigned_constraint = [[], []]
			for j in V_prime:
				passenger_assigned_constraint[0].append('x({0},{1})'.format(i.num, j.num))
				passenger_assigned_constraint[1].append(1)
				
			constraint_names.append('passenger_{0}_assigned_constraint'.format(i.num))
			constraints.append(passenger_assigned_constraint)
			constraint_rhs.append(1)
			constraint_sense.append('E')
			
		#only one reassignment
		for i in R_joined:
			for j in V_prime:
				one_standard_reassignment_constraint = [[], []]
				one_standard_reassignment_constraint[0].append('x({0},{1})'.format(i.num, j.num))
				one_standard_reassignment_constraint[1].append(b[i.num])
			
				constraint_names.append('passenger_{0}_vehicle_{1}_one_reassignment'.format(i.num, j.num))
				constraints.append(one_standard_reassignment_constraint)
				constraint_rhs.append(b[i.num] * y[i.num][j.num])
				constraint_sense.append('G')

		#makes sure we don't kick passengers out of cars they're already in
		#may need to generate a separate list of xij and x'ij lists to make sure that this is guaranteed to not change; working off comparisons with y list may not work
		V_joined = set().union(V_D)
		for vehicle in V_P:
			if len(V_P) == 2:
				V_joined.add(vehicle)

		# for i in R_IV:
		# 	for j in V_joined:
		# 		no_kick_out_constraint_x = [[], []]
		# 		no_kick_out_constraint_x[0].append('x({0},{1})'.format(i.num, j.num))
		# 		no_kick_out_constraint_x[1].append(1)
		# 		constraint_names.append('passenger_{0}_in_vehicle_{1}_no_swap_x'.format(i.num, j.num))
		# 		constraints.append(no_kick_out_constraint_x)
		# 		constraint_rhs.append(x_prev[i.num][j.num])
		# 		constraint_sense.append('E')
		# 		print(no_kick_out_constraint_x, i.num, j.num, x_prev[i.num][j.num])


		problem.variables.add(obj = obj, lb = lb, ub = ub, names = names, types = variable_types)
		problem.linear_constraints.add(lin_expr = constraints, senses = constraint_sense, rhs = constraint_rhs, names = constraint_names)

		problem.solve()
		
		values = problem.solution.get_values()
		# for i in range(len(names)):
		# 	if 'x(' in names[i]:
		# 		print(names[i] + '      : ' + str(values[i]))
		# 	else:
		# 		print(names[i] + ': ' + str(values[i]), x_prime_prev)

		print("Total Cost = " + str(problem.solution.get_objective_value()))
		
		#updates sets of assigned passengers and vehicles for the next time step

		#reconfigures vehicle and passenger info due to reassignments
		for ind in range(len(names)):

			print(str(names[ind]), '      : ' , str(values[ind]))


			
		
	#set constants
	phi = drop_off_pen
	delta = reassign_pen
	gamma = wait_pen
	psi1 = pass1_pen
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

	solve_R_lessthan_V(R, R_A, R_IV, R_prime, V, V_P, V_prime, t)

	#loop
	# while t <= T:
	# 	print('Modeling time = ' + str(t))
	# 	for passenger in R:
	# 		print('Passenger #' + str(passenger.num) + ' pos: (' + str(passenger.x) + ',' + str(passenger.y) + ')   dest: ', passenger.d, '   dist: ', point_dist((passenger.x, passenger.y), passenger.d), '  appearing ', passenger.appear)
	# 	for vehicle in V:
	# 		print('Vehicle #' + str(vehicle.num) + ' pos: (' + str(vehicle.x) + ',' + str(vehicle.y) + ')')
		
	# 	# update_in_vehicle(R, R_IV, R_S, V_I, V_D)
	# 	# update_assigned(R, R_A, R_IV, V_P, V_D)

	# 	print('POST')
	# 	print('R_U' , R_U)
	# 	print('R_A' , R_A)
	# 	print('R_prime', R_prime)

	# 	print()

	# 	print('R_IV', R_IV)
	# 	print('R_S' , R_S)

	# 	print()

	# 	print('V_I' , V_I)
	# 	print('V_P' , V_P)
	# 	print('V_D' , V_D)
	# 	print('V_prime', V_prime)

	# 	# for vehicle in V:
	# 	# 	for p in vehicle.passengers:
	# 	# 		print(vehicle, ' ', vehicle.num, ' is serving ', R[p], ' ', R[p].num)
	# 	# 	if vehicle.next is not None:
	# 	# 		print(vehicle, ' ', vehicle.num, ' will eventually pick up ', R[p], ' ', R[p].num)

	# 	solve_R_lessthan_V(R, R_A, R_IV, R_prime, V, V_P, V_prime, t)
	# 	t += t_int
	# 	for passenger in R:
	# 		if passenger.appear <= t and passenger.appear > t - t_int:
	# 			R_prime.add(passenger)
	# 			R_U.add(passenger)

	# 	if len(R_S) == len(R):
	# 		break

	print('finished')

class Passenger:
	def __init__(self, num, x_max, y_max, time_horizon):
		self.num = num
		self.x = random.random() * x_max
		self.y = random.random() * y_max
		self.o = (self.x, self.y)
		self.d = (round(random.random() * x_max, 2), round(random.random() * y_max, 2))
		self.state = 'unassigned'
		self.vehicle = None #vehicle number that is serving it; not the vehicle object!
		self.reassigned = 0
		self.wait = 0
		self.appear = random.random() * time_horizon * 0.3
		self.appear = 0 #for testing

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

#Choose inputs here:
number_of_passengers = 10
number_of_vehicles = 10
vehicle_speed = 60. #kmh 55 default
x_size = 10. #km
y_size = 10. #km
run_horizon = 1 #hours
update_interval = 1 #seconds
dropoff_reasssignment_penalty = 1.
reassignment_penalty = 250000. #km * seconds
waiting_penalty = 20000. #km/seconds
pass1_distance_pen = 0.8

#simulation is calculated in km and seconds
vehicle_speed /= 3600. #kms
run_horizon *= 3600. #s

simulate_rideshare(number_of_passengers, number_of_vehicles, vehicle_speed, x_size, y_size, run_horizon, update_interval, dropoff_reasssignment_penalty, reassignment_penalty, waiting_penalty, pass1_distance_pen)
