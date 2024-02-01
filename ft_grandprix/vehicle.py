from dataclasses import dataclass

@dataclass
class VehicleStateSnapshot:
    laps : int
    velocity : list
    yaw : float
    pitch : float
    roll : float
    lap_completion : int
    absolute_completion : int
    time : float
