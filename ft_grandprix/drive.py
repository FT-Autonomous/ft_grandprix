import time
import argparse
import mujoco
import mujoco.viewer
import json
import os
import numpy as np
from .fast import Driver

# Define the control actions for the car
control_actions = {
    265: [1.0, 0.0],  # Move the car forward
    264: [-1.0, 0.0],  # Move the car backward
    263: [0.0, 1.0],  # Turn the car left
    262: [0.0, -1.0],  # Turn the car right
}

# dpg.create_context()

class Simulator:
    def __init__(self, map_dir="rendered"):
        mjcf_metadata_path = os.path.join(map_dir, "car.json")
        model_path = os.path.join(map_dir, "car.xml")
        
        with open(mjcf_metadata_path) as mjcf_metadata_file:
            mjcf_metadata = json.load(mjcf_metadata_file)
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.paused = False
        self.ctrl_indices = []
        self.sensor_indices = []
        self.drivers = []
        self.watching = 0
        
        for i in range(len(mjcf_metadata["cars"])):
            indices = []
            for j in range(mjcf_metadata["rangefinders"]):
                sensor = self.model.sensor(f"rangefinder #{i}.#{j}")
                indices.append(sensor.id)
            self.sensor_indices.append(np.array(indices))
            forward = self.model.actuator(f"forward #{i}")
            turn = self.model.actuator(f"turn #{i}")
            self.ctrl_indices.append([forward.id, turn.id])
            self.drivers.append(Driver())
        
    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.paused = not self.paused
            print("Paused:", self.paused)
        elif chr(keycode) == "P":
            self.watching = ((self.watching or 0) - 1) % len(self.drivers)
        elif chr(keycode) == "N":
            self.watching = ((self.watching or 0) + 1) % len(self.drivers)
        elif chr(keycode) == "L":
            self.watching = None
        elif keycode in control_actions:
            # print("fuckk")
            # control_action = control_actions[keycode]
            # print(self.data.ctrl[:], control_action)
            # print("\n\n")
            # self.data.ctrl[:] = control_action
            # print("Control Action:", keycode)
            pass
        else:
            print(keycode)

    def drive(self, map_dir="rendered"):
        with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self.key_callback
        ) as viewer:
            while True:
                for driver, sensors, (forward, turn) in zip(
                        self.drivers,
                        self.sensor_indices,
                        self.ctrl_indices
                ):
                    speed, steering_angle = driver.process_lidar(self.data.sensordata[sensors])
                    self.data.ctrl[forward] = speed
                    self.data.ctrl[turn] = steering_angle
                step_start = time.time()
            
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                if not self.paused:
                    mujoco.mj_step(self.model, self.data)

                with viewer.lock():
                    if self.watching is not None:
                        viewer.cam.lookat[:] = self.data.body(f"car #{self.watching}").xpos[:]
                    # breakpoint()
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
            
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-dir",
        help="the directory containing rendered xml and json files from ft_grandprix.map",
        default="rendered",
        dest="map_dir"
    )
    args = parser.parse_args()
    Simulator(**args.__dict__).drive()

if __name__ == "__main__":
    main()
