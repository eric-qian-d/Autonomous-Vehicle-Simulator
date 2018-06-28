\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: 6.33995118625078 x(1,0) + 31.7615449433431 x(1,1) + 25 x_prime(1,1)
      + 31.5108383029782 x(0,0) + 25 x_prime(0,0) + 0.0727743926548009 x(0,1)
Subject To
 vehicle_0_no_2_x_constraint:    x(1,0) + x(0,0) <= 1
 vehicle_1_no_2_x_constraint:    x(1,1) + x(0,1) <= 1
 initial_rider_constraint_0:     x_prime(1,0) + x_prime(0,0) <= 0
 single_ride_cap_constraint_0:   x(1,0) + x(0,0) <= 1
 shared_ride_cap_constraint_0:   x_prime(1,0) + x_prime(0,0) <= 1
 initial_rider_constraint_1:     x_prime(1,1) + x_prime(0,1) <= 0
 single_ride_cap_constraint_1:   x(1,1) + x(0,1) <= 1
 shared_ride_cap_constraint_1:   x_prime(1,1) + x_prime(0,1) <= 1
 passenger_1_assigned_constraint: x(1,0) + x_prime(1,0) + x(1,1) + x_prime(1,1)
                                  = 1
 passenger_0_assigned_constraint: x(0,0) + x_prime(0,0) + x(0,1) + x_prime(0,1)
                                  = 1
 passenger_1_vehicle_0_one_reassignment: >= 0
 passenger_1_vehicle_1_one_reassignment: >= 0
 passenger_0_vehicle_0_one_reassignment: >= 0
 passenger_0_vehicle_1_one_reassignment: >= 0
 passenger_1_vehicle_0_forced_next_pickup_constraint: x(1,0) <= 1
 passenger_1_vehicle_1_forced_next_pickup_constraint: x(1,1) <= 1
 passenger_0_vehicle_0_forced_next_pickup_constraint: x(0,0) <= 1
 passenger_0_vehicle_1_forced_next_pickup_constraint: x(0,1) <= 1
Bounds
 0 <= x(1,0) <= 1
 0 <= x_prime(1,0) <= 1
 0 <= x(1,1) <= 1
 0 <= x_prime(1,1) <= 1
 0 <= x(0,0) <= 1
 0 <= x_prime(0,0) <= 1
 0 <= x(0,1) <= 1
 0 <= x_prime(0,1) <= 1
End
