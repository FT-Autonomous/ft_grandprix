# TODO: only render on user input when the thing is paused
# TODO: remove all python to avoid GIL
# TODO: dont store car state in 10 billion different arrays

import os
import math
import importlib
import json
import mujoco
import numpy as np
import time
import dearpygui.dearpygui as dpg
from .nidc import Driver
import threading
from .curve import extract_path_from_svg

np.set_printoptions(
    precision=2,
    formatter={"float": lambda x: f"{x:8.2f}"}
)

def quaternion_to_angle(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return yaw_z

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    return [qw, qx, qy, qz]

def scatter(path):
    import matplotlib.pyplot as plt
    for i in range(len(path)):
        x = path[i][0]
        y = path[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
    plt.show()

width, height = 640, 480
pixels = np.zeros((height, width, 3), dtype=np.float32)
exit_event = threading.Event()

class Real:
    def __init__(self, /, meta):
        self.meta = meta
        self.completion_id = dpg.generate_uuid()
        self.laps_id = dpg.generate_uuid()
        self.times_id = dpg.generate_uuid()
        self.reload_code_id = dpg.generate_uuid()

class Meta:
    # TODO: destructure this into mutable current lap data and other
    # persistent / append only data
    def __init__(self, /, id, forward, turn, sensors, driver, label):
        self.id = id
        self.completion = 0 # current completion percentage
        self.ready = False # if we are ready for a new lap, set once we get half way around
        self.start = 0 # physics time step we are currently at
        self.laps = 0 # number of laps completed
        self.times = [] # lap times
        self.forward = forward
        self.turn = turn
        self.sensors = sensors
        self.driver = driver
        self.label = label

# Only reason this exists is so that combos can have supplemental
# data?
class Pun:
    def __init__(self, data, inner):
        self.data = data
        self.inner = inner
    def __repr__(self):
        return self.data
    
class ModelAndView:
    # inject a list of meta objects into the view
    def inject_meta(self, meta):
        meta = { m.id : m for m in meta } # payload being injected
        for id in dpg.get_item_children("D", 1):
            real = dpg.get_item_configuration(id)['user_data']
            m = real.meta
            if m.id not in meta:
                dpg.delete_item(id)
            else:
                del meta[m.id]
                dpg.set_value(real.completion_id, str(m.completion) + "%")
                dpg.set_value(real.laps_id, str(m.laps))
                dpg.set_value(real.times_id, str(m.times))
        for m in meta:
            m = meta[m]
            real = Real(m)
            with dpg.collapsing_header(user_data=real, parent="D", label=f"Car #{m.id} - {m.label}", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Completion:")
                    dpg.add_text("0%", tag=real.completion_id)
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Laps:")
                    dpg.add_text("0", tag=real.laps_id)
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Laps Times:")
                    dpg.add_text("0", tag=real.times_id)
                dpg.add_button(label="Reload Code", tag=real.reload_code_id, user_data=real)
            
        with dpg.mutex():
            pass
    
    def __init__(self):
        self.mj = Mujoco(self) # our model
        self.paused = False
        self.last = None
        self.pause()

    def pause(self):
        self.paused = not self.paused
        if self.paused:
            self.mj.running_event.clear()
        else:
            self.mj.running_event.set()

    def reset(self):
        self.mj.reset_event.set()

    def scroll(self, sender, value):
        self.mj.camera.distance -= value / 20
        
    def release(self):
        self.last = None

    def drag(self, sender, value):
        [x, y] = dpg.get_mouse_pos(local=False)
        delta = [dx, dy] = dpg.get_mouse_drag_delta()
        if self.last is not None:
            dx -= self.last[0]
            dy -= self.last[1]
            self.last = delta
        else:
            self.last = delta
        state = dpg.get_item_state('simulation')
        if state['rect_min'][0] <= x < state['rect_max'][0] \
           and state['rect_min'][1] < y <= state['rect_max'][1]:
                self.mj.camera.azimuth -= dx
                self.mj.camera.elevation -= dy

    def rangefinder(self, sender, value):
        self.mj.model.vis.rgba.rangefinder[3] = value

    def release_key(self, sender, keycode):
        if chr(keycode) == " ":
            self.pause()
        elif chr(keycode) == "P":
            self.mj.watching = ((self.mj.watching or 0) - 1) % len(self.mj.meta)
        elif chr(keycode) == "N":
            self.mj.watching = ((self.mj.watching or 0) + 1) % len(self.mj.meta)
        elif chr(keycode) == "L":
            self.mj.watching = None
        elif chr(keycode) == "S":
            self.mj.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] ^= 1
        else:
            print(keycode)
        
    def run(self):
        print("GUI thread spawned")
        dpg.create_context()
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.scroll)
            dpg.add_mouse_release_handler(callback=self.release)
            dpg.add_mouse_drag_handler(callback=self.drag)
            dpg.add_key_release_handler(callback=self.release_key)
        with dpg.texture_registry(show=False):
            global pixels
            dpg.add_raw_texture(width, height, pixels, format=dpg.mvFormat_Float_rgb, tag="texture_tag")
        with dpg.window(tag="Main"):
            with dpg.group(tag="Main Group", horizontal=True):
                with dpg.group(tag="Visualisation"):
                    dpg.add_image("texture_tag", tag="simulation")
                dpg.add_spacer(width=10)
                with dpg.group(tag="Settings", width=200):
                    dpg.add_button(label="Reset Simulation", callback=self.reset)
                    dpg.add_button(label="Pause Simulation", callback=self.pause)
                    dpg.add_combo(default_value="track", # TODO: make sure that this actually corresponds to the value set in launch.py
                                  items=["track", "blank"], label="map", callback=self.select_map_callback)
                    dpg.add_separator()
                    
                    with dpg.group(tag="D"):
                        pass
                    
                    dpg.add_separator()
                    dpg.add_slider_float(label="rangefinder Intensity", default_value=0.1, max_value=0.3, callback=self.rangefinder)
        dpg.bind_item_handler_registry("simulation", "bem")
        dpg.create_viewport(title='Custom Title', width=1124, height=612)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Main", True)
        dpg.start_dearpygui()
        
    def select_map_callback(self, sender, value):
        print(value)
        pass

class Mujoco:
    def __init__(self, mv):
        self.reset_event = threading.Event()
        self.running_event = threading.Event()
        self.running_event.set()
        self.paused = False
        self.mv = mv
        
        print("Running mujoco thread")
        map_dir = "rendered"
        mjcf_metadata_path = os.path.join(map_dir, "car.json")
        map_metadata_path = os.path.join(map_dir, "chunks/metadata.json")
        with open(map_metadata_path) as map_metadata_file:
            self.map_metadata = json.load(map_metadata_file)
        model_path = os.path.join(map_dir, "car.xml")
        with open(mjcf_metadata_path) as mjcf_metadata_file:
            self.mjcf_metadata = json.load(mjcf_metadata_file)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        self.watching = 0
        self.meta = []
        
        for i, car in enumerate(self.mjcf_metadata["cars"]):
            indices = []
            for j in range(self.mjcf_metadata["rangefinders"]):
                sensor = self.model.sensor(f"rangefinder #{i}.#{j}")
                indices.append(sensor.id)
            if car["driver"].startswith("file://"):
                path = car["driver"][7:-3].replace("/", ".")
                print(f"Loading driver from path '{path}'")
                spec = importlib.util.find_spec(path)
                driver = spec.loader.load_module().Driver()
            else:
                print("Unsupported schema: supported (file://)")
            meta = Meta(
                id      = i,
                forward = self.model.actuator(f"forward #{i}").id,
                turn    = self.model.actuator(f"turn #{i}").id,
                sensors = np.array(indices),
                driver  = driver,
                label   = car["name"]
            )
            self.meta.append(meta)
            
        self.camera = mujoco.MjvCamera()
        self.physics = threading.Thread(target=self.physics_thread)
        self.physics.start()
        self.render = threading.Thread(target=self.render_thread)
        self.render.start()

    def position_vehicles(self, path):
        # TODO: accomodate for the fact that cars start off at
        # different locations in lap time / completion computations

        # TODO: make sure that we can function without a
        # path. E.g. for testing maps or for when people make maps
        # without an SVG.
        
        for i in range(len(self.meta)):
            joint = self.data.joint(f"car #{i}")
            i = (i+5)*2
            delta = path[i+1]-path[i]
            angle = math.atan2(delta[1], delta[0])
            quat = euler_to_quaternion([angle, 0, 0])
            joint.qpos[3:] = np.array(quat)
            joint.qpos[:2] = path[i]

    def physics_thread(self):
        global exit_event
        last = time.time()
        path = extract_path_from_svg("template/path1-regular.svg")
        path[:, 0] =   path[:, 0] / self.map_metadata['width']  * 20
        path[:, 1] = - path[:, 1] / self.map_metadata['height'] * 20
        physics_steps = 0
        self.position_vehicles(path)
        while True:
            if self.watching is not None:
                self.camera.lookat[:] = self.data.body(f"car #{self.watching}").xpos[:]
            if self.reset_event.is_set():
                mujoco.mj_resetData(self.model, self.data)
                self.position_vehicles(path)
                self.reset_event.clear()
                # print("RESET")
            if exit_event.is_set():
                break
            self.running_event.wait(timeout=0.25)
            if not self.running_event.is_set():
                mujoco.mj_forward(self.model, self.data)
                continue
            for index, meta in enumerate(self.meta):
                xpos = self.data.joint(f"car #{index}").qpos[0:2]
                [x, y] = list(xpos)
                completion = ((path - xpos)**2).sum(1).argmin()
                if 45 <= completion <= 55:
                    meta.ready = True
                delta = abs(completion - meta.completion)
                if meta.ready and delta > 90:
                    meta.times.append((physics_steps - meta.start) * 0.005) # TODO: change this constant to always reflect physics timestep
                    meta.ready = False
                    meta.laps += 1
                    meta.start = physics_steps
                meta.completion = completion

                id = self.model.sensor("car #0 accelerometer").id
                d = self.data.sensordata[id:id+3]
                # print(f"ACCEL: data: {d} accel: {math.sqrt((d**2).sum())}")
                id = self.model.sensor("car #0 gyro").id
                d = self.data.sensordata[id:id+3]
                # print(f"GYRO:  accel: {math.sqrt((d**2).sum())}")

                speed, steering_angle = meta.driver.process_lidar(self.data.sensordata[meta.sensors])
                self.data.ctrl[meta.forward] = speed
                self.data.ctrl[meta.turn] = steering_angle
                # print (speed, steering_angle / 3.14)
            mujoco.mj_step(self.model, self.data)
            physics_steps += 1
            
            now = time.time()
            fps = 1 / (now - last)
            last = time.time()
            # print(f"{fps} fps")

        
    def render_thread(self):
        global exit_event
        global pixels
        global width
        global height
        last = time.time()
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        buf = np.zeros((height, width, 3), dtype=np.uint8)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        max_fps = 5
        while True:
            # make it so that you can reset while paused …
            # self.running_event.wait()
            
            self.mv.inject_meta(self.meta)
            self.renderer.update_scene(self.data, self.camera)
            self.renderer.render(out=buf)
            pixels[...] = buf / 255.0
            now = time.time()
            if (now - last < 1/max_fps):
                time.sleep(1/max_fps - (now - last));
            # print (f"{1/(time.time()-last)} fps")
            last = time.time()
            if exit_event.is_set(): break

if __name__ == "__main__":
    try:
        ModelAndView().run()
    except KeyboardInterrupt:
        pass
    exit_event.set()
