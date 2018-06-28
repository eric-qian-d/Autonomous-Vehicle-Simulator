\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: - 0.108145596470039 x(0,0) + x_prime(0,0) + 0.736913588240646 x(1,0)
      + 2 x_prime(1,0)
Subject To
 vehicle_0_no_2_x_constraint:    x(0,0) + x(1,0) <= 1
 initial_rider_constraint_0:     x_prime(0,0) + x_prime(1,0) <= 0
 single_ride_cap_constraint_0:   x(0,0) + x(1,0) <= 1
 shared_ride_cap_constraint_0:   x_prime(0,0) + x_prime(1,0) <= 1
 passenger_0_assigned_constraint: x(0,0) + x_prime(0,0) <= 1
 passenger_1_assigned_constraint: x(1,0) + x_prime(1,0) <= 1
 passenger_1_no_become_unassigned:  = 1
 max_seats_used_constraint:      x(0,0) + x_prime(0,0) + x(1,0) + x_prime(1,0)
                                  = 1
 passenger_0_vehicle_0_one_reassignment: >= 0
 passenger_1_vehicle_0_one_reassignment: >= 0
 passenger_0_vehicle_0_forced_next_pickup_constraint: x(0,0) <= 1
 passenger_1_vehicle_0_forced_next_pickup_constraint: x(1,0) <= 1
Bounds
 0 <= x(0,0) <= 1
 0 <= x_prime(0,0) <= 1
 0 <= x(1,0) <= 1
 0 <= x_prime(1,0) <= 1
End
