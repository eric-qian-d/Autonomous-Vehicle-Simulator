import simulation as sim
import csv
	
passengers_used = []
vehicles_used = []
vehicle_speed_used = []
sim_x_size_used = []
sim_y_size_used = []
run_horizon_used = []
update_interval_used = []
dropoff_reasssignment_penalty_used = []
reassignment_penalty_used = []
waiting_penalty_used = []
pass1_distance_pen_used = []
pass2_distance_pen_used = []
rideshare_flat_penalty_used = []
rideshare_allowed = []

variables = [passengers_used, vehicles_used, vehicle_speed_used, sim_x_size_used, sim_y_size_used, run_horizon_used, 
			update_interval_used, dropoff_reasssignment_penalty_used, reassignment_penalty_used, waiting_penalty_used,
			pass1_distance_pen_used, pass2_distance_pen_used, rideshare_flat_penalty_used, rideshare_allowed]

run_time = []
num_served = []
empty_km = []


results = [run_time, num_served, empty_km]

all_runs = []

# for passengers in range(50, 100):
# 		for vehicles in range(20, 40):
# 			for rideshare_used in range(2):
# 				number_of_passengers = passengers
# 				number_of_vehicles = vehicles
# 				vehicle_speed = 60. #kmh 55 default
# 				x_size = 10. #km
# 				y_size = 10. #km
# 				run_horizon = 3. #hours
# 				update_interval = 10. #seconds
# 				dropoff_reasssignment_penalty = 1 #km
# 				reassignment_penalty = 1. #km * seconds
# 				waiting_penalty = .05 #km/seconds
# 				pass1_distance_pen = 1.5
# 				pass2_distance_pen = 1.1
# 				rideshare_flat_penalty = 1.8 #km
# 				rideshare = True if rideshare_used == 1 else False

# 				#simulation is calculated in km and seconds
# 				vehicle_speed /= 3600. #kms
# 				run_horizon *= 3600. #s

# 				try:
# 					time, served, empty = sim.simulate_rideshare(number_of_passengers, number_of_vehicles, vehicle_speed, x_size, y_size, run_horizon, update_interval, dropoff_reasssignment_penalty, reassignment_penalty, waiting_penalty, pass1_distance_pen, pass2_distance_pen, rideshare_flat_penalty, rideshare)
# 					# print('blah')
					
# 					# print('testing here', time, served, empty)

# 					passengers_used.append(number_of_passengers)
# 					vehicles_used.append(number_of_vehicles)
# 					vehicle_speed_used.append(vehicle_speed)
# 					sim_x_size_used.append(x_size)
# 					sim_y_size_used.append(y_size)
# 					run_horizon_used.append(run_horizon)
# 					update_interval_used.append(update_interval)
# 					dropoff_reasssignment_penalty_used.append(dropoff_reasssignment_penalty)
# 					reassignment_penalty_used.append(reassignment_penalty)
# 					waiting_penalty_used.append(waiting_penalty)
# 					pass1_distance_pen_used.append(pass1_distance_pen)
# 					pass2_distance_pen_used.append(pass2_distance_pen)
# 					rideshare_flat_penalty_used.append(rideshare_flat_penalty)
# 					rideshare_allowed.append(rideshare)
# 					run_time.append(time)
# 					num_served.append(served)
# 					empty_km.append(empty)
# 				except Exception:
# 					print('broken')
# 					with open ('csvfile.csv','a') as file:
# 						writer = csv.writer(file)
# 						temp = ['passengers', 'vehicles', 'vehicle speed', 'x-dim', 'y-dim', 'time horizon', 'update interval', 'immediate pickup penalty', 
# 						'reassignment penalty', 'waiting penalty', 'pass1 penalty', 'pass2 penalty', 'rideshare flat penalty', 'rideshares', 'run time', 'number served', 'empty km']
# 						writer.writerow(temp)

# 						for i in range(len(variables[0])):
# 							res = []
# 							for j in range(len(variables)):
# 								res.append(str(variables[j][i]))
# 							for j in range(len(results)):
# 								res.append(str(results[j][i]))
# 							writer.writerow(res)
# 					break
p1 = 0.5
p2 = 0.5
flat = 0.5


while p1 <= 3:
		while p2 <= 3:
			while flat <= 3:
				number_of_passengers = 81
				number_of_vehicles = 33
				vehicle_speed = 60. #kmh 55 default
				x_size = 10. #km
				y_size = 10. #km
				run_horizon = 3. #hours
				update_interval = 10. #seconds
				dropoff_reasssignment_penalty = 1 #km
				reassignment_penalty = 1. #km * seconds
				waiting_penalty = .05 #km/seconds
				pass1_distance_pen = 1.5
				pass2_distance_pen = 1.1
				rideshare_flat_penalty = 1.8 #km
				rideshare = True if rideshare_used == 1 else False

				#simulation is calculated in km and seconds
				vehicle_speed /= 3600. #kms
				run_horizon *= 3600. #s


				p1 += 0.1
				p2 += 0.1
				flat += 0.1

				try:
					time, served, empty = sim.simulate_rideshare(number_of_passengers, number_of_vehicles, vehicle_speed, x_size, y_size, run_horizon, update_interval, dropoff_reasssignment_penalty, reassignment_penalty, waiting_penalty, pass1_distance_pen, pass2_distance_pen, rideshare_flat_penalty, rideshare)
					# print('blah')
					
					# print('testing here', time, served, empty)

					passengers_used.append(number_of_passengers)
					vehicles_used.append(number_of_vehicles)
					vehicle_speed_used.append(vehicle_speed)
					sim_x_size_used.append(x_size)
					sim_y_size_used.append(y_size)
					run_horizon_used.append(run_horizon)
					update_interval_used.append(update_interval)
					dropoff_reasssignment_penalty_used.append(dropoff_reasssignment_penalty)
					reassignment_penalty_used.append(reassignment_penalty)
					waiting_penalty_used.append(waiting_penalty)
					pass1_distance_pen_used.append(pass1_distance_pen)
					pass2_distance_pen_used.append(pass2_distance_pen)
					rideshare_flat_penalty_used.append(rideshare_flat_penalty)
					rideshare_allowed.append(rideshare)
					run_time.append(time)
					num_served.append(served)
					empty_km.append(empty)
				except Exception:
					print('broken')
					with open ('csvfile.csv','a') as file:
						writer = csv.writer(file)
						temp = ['passengers', 'vehicles', 'vehicle speed', 'x-dim', 'y-dim', 'time horizon', 'update interval', 'immediate pickup penalty', 
						'reassignment penalty', 'waiting penalty', 'pass1 penalty', 'pass2 penalty', 'rideshare flat penalty', 'rideshares', 'run time', 'number served', 'empty km']
						writer.writerow(temp)

						for i in range(len(variables[0])):
							res = []
							for j in range(len(variables)):
								res.append(str(variables[j][i]))
							for j in range(len(results)):
								res.append(str(results[j][i]))
							writer.writerow(res)
					break

print(results)

with open ('csvfile.csv','a') as file:
	writer = csv.writer(file)
	temp = ['passengers', 'vehicles', 'vehicle speed', 'x-dim', 'y-dim', 'time horizon', 'update interval', 'immediate pickup penalty', 
	'reassignment penalty', 'waiting penalty', 'pass1 penalty', 'pass2 penalty', 'rideshare flat penalty', 'rideshares', 'run time', 'number served', 'empty km']
	writer.writerow(temp)

	for i in range(len(variables[0])):
		res = []
		for j in range(len(variables)):
			res.append(str(variables[j][i]))
		for j in range(len(results)):
			res.append(str(results[j][i]))
		writer.writerow(res)

print('worked')