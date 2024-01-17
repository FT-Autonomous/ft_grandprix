from dataclasses import dataclass

@dataclass
class VehicleStateSnapshot:
    laps : int
    velocity : list[float]
    lap_completion : int
    absolute_completion : int
