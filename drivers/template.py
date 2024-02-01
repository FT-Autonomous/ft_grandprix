class Driver:
    def process_lidar(self, ranges, state):
        """
        Produces controls for the car based on sensor input

        Args:
        
            ranges (array): the distances of the LiDAR measurements
            from the bottom of the car counterclockwise around to the
            front of the car.

            state (VehicleStateSnapshot): metadata about the current state
            of the vehicle in the world, such as speed and orientation.
        
        """
        speed = 0
        steering_angle = 0
        return speed, steering_angle
