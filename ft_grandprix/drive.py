import time
import mujoco
import mujoco.viewer
import json
import numpy as np
from .nidc import Driver

with open("models/car.json") as mjcf_metadata_file:
    mjcf_metadata = json.load(mjcf_metadata_file)

m = mujoco.MjModel.from_xml_path("models/car.xml")
d = mujoco.MjData(m)

# Define the control actions for the car
control_actions = {
    265: [1.0, 0.0],  # Move the car forward
    264: [-1.0, 0.0],  # Move the car backward
    263: [0.0, 1.0],  # Turn the car left
    262: [0.0, -1.0],  # Turn the car right
}

paused = False

ctrl_indices = []
sensor_indices = []
drivers = []
watching = 0

for i in range(len(mjcf_metadata["cars"])):
    indices = []
    for j in range(mjcf_metadata["rangefinders"]):
        sensor = m.sensor(f"rangefinder #{i}.#{j}")
        indices.append(sensor.id)
    sensor_indices.append(np.array(indices))
    forward = m.actuator(f"forward #{i}")
    turn = m.actuator(f"turn #{i}")
    ctrl_indices.append([forward.id, turn.id])
    drivers.append(Driver())

def key_callback(keycode):
    global paused
    global drivers
    global watching
    if chr(keycode) == " ":
        paused = not paused
        print("Paused:", paused)
    elif chr(keycode) == "P":
        watching = (watching - 1) % len(drivers)
    elif chr(keycode) == "N":
        watching = (watching + 1) % len(drivers)
    elif keycode in control_actions:
        # print("fuckk")
        # control_action = control_actions[keycode]
        # print(d.ctrl[:], control_action)
        # print("\n\n")
        # d.ctrl[:] = control_action
        # print("Control Action:", keycode)
        pass
    else:
        print(keycode)


with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    while True:
        for driver, sensors, (forward, turn) in zip(drivers, sensor_indices, ctrl_indices):
            speed, steering_angle = driver.process_lidar(d.sensordata[sensors])
            d.ctrl[forward] = speed
            d.ctrl[turn] = steering_angle / 1.5
        step_start = time.time()
    
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        if not paused:
            mujoco.mj_step(m, d)

        with viewer.lock():
            viewer.cam.lookat[:] = d.body(f"car #{watching}").xpos[:]
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
    
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
