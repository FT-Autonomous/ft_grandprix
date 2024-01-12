# TODO: only render on user input when the thing is paused

import os
import math
import importlib
import queue
import json
import mujoco
import numpy as np
import time
import dearpygui.dearpygui as dpg
from .nidc import Driver
import threading
from .curve import extract_path_from_svg
from .chunk import chunk
from .map import produce_mjcf
from .vendor import Renderer

np.set_printoptions(precision=2, formatter={"float": lambda x: f"{x:8.2f}"})

def runtime_import(path):
    spec = importlib.util.find_spec(path)
    module =  spec.loader.load_module()
    return module

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


exit_event = threading.Event()

class Real:
    def __init__(self, /, id):
        self.id = id
        self.completion_id = dpg.generate_uuid()
        self.laps_id = dpg.generate_uuid()
        self.times_id = dpg.generate_uuid()

class Meta:
    # TODO: destructure this into mutable current lap data and other
    # persistent / append only data
    def __init__(self, /, id, offset, driver, label, driver_path):
        self.id = id
        self.offset = offset
        self.completion = 0 # current completion percentage
        self.ready = False # if we are ready for a new lap, set once we get half way around
        self.start = 0 # physics time step we are currently at
        self.laps = 0 # number of laps completed
        self.times = [] # lap times
        self.driver = driver
        self.driver_path = driver_path
        self.label = label

        # lifetime of model/data (use reload_mujoco after construction)
        self.forward = None
        self.turn = None
        self.sensors = None
        self.joint = None
        
    def reload_mujoco(self, data, /, rangefinders):
        self.forward     = data.actuator(f"forward #{self.id}").id
        self.turn        = data.actuator(f"turn #{self.id}").id
        self.joint       = data.joint(f"car #{self.id}")
        self.sensors     = [data.sensor(f"rangefinder #{self.id}.#{j}").id for j in range(rangefinders)]

    def reload_code(self):
        self.driver = runtime_import(self.driver_path).Driver()
    
class ModelAndView:
    def __init__(self, track, width=1124, height=612):
        self.window_size = width, height
        self.pixels = np.zeros((1080, 1920, 3), dtype=np.float32)
        self.mj = Mujoco(self, track=track) # our model
        self.last = None

    def simulation_viewport_size(self):
        return [max(self.window_size[0] - 400, 0),
                max(self.window_size[1] - 20, 0)]
    
    def inject_meta(self, meta):
        meta = set(meta) # payload being injected
        for id in dpg.get_item_children("Dashboard", 1):
            real = dpg.get_item_configuration(id)['user_data']
            m = self.mj.meta[real.id]
            if m.id not in meta:
                dpg.delete_item(id)
            else:
                if real.id in meta: meta.remove(real.id)
                dpg.set_value(real.completion_id, str(m.completion) + "%")
                dpg.set_value(real.laps_id, str(m.laps))
                dpg.set_value(real.times_id, str(m.times))
        for m in meta:
            real = Real(m)
            m = self.mj.meta[m]
            with dpg.collapsing_header(user_data=real, parent="Dashboard", label=f"Car #{m.id} - {m.label}", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Completion:")
                    dpg.add_text("0%", tag=real.completion_id)
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Laps:")
                    dpg.add_text("0", tag=real.laps_id)
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Lap Times:")
                    dpg.add_text("0", tag=real.times_id)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reload Code", user_data=real.id, callback=self.reload_code_cb)
                    dpg.add_button(label="Info", user_data=real.id)

    def reload_code_cb(self, sender, value, user_data):
        if dpg.does_item_exist("reload code modal"):
            dpg.delete_item("reload code modal")
        car_index = user_data
        try:
            self.mj.reload_code(car_index)
        except Exception as e:
            with dpg.window(tag="reload code modal", modal=True, show=True) as window:
                dpg.add_text(str(e), color=[255, 200, 200])
                dpg.add_button(label="Try Again", callback=self.reload_code_cb, user_data=car_index)
    
    def pause(self):
        if self.mj.running_event.is_set():
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
        state = dpg.get_item_state('simulation')
        [x, y] = dpg.get_mouse_pos(local=False)
        bounded = \
            state['rect_min'][0] <= x < state['rect_max'][0] \
            and state['rect_min'][1] < y <= state['rect_max'][1]
        if not bounded:
            return
            
        delta = [dx, dy] = dpg.get_mouse_drag_delta()
        if self.last is not None:
            dx -= self.last[0]
            dy -= self.last[1]
            self.last = delta
        else:
            self.last = delta
        if not self.mj.cinematic:
            self.mj.camera.azimuth   += dx
            self.mj.camera.elevation += dy
        else:
            self.mj.camera_vel[0] += dx / 100
            self.mj.camera_vel[1] += dy / 100
        
    def rangefinder(self, sender, value):
        self.mj.model.vis.rgba.rangefinder[3] = value

    def release_key(self, sender, keycode):
        # TODO: Rewrite this so that help can be autogenerated. Each setting
        # would be a closure that captures this class and implements apply(),
        # description() and status(). Each closure would then be associated
        # with a key externally.
        
        if chr(keycode) == " ":
            self.pause()
        elif chr(keycode) == "C":
            self.mj.cinematic = not self.mj.cinematic
            if not self.mj.cinematic:
                self.mj.camera_vel[0] = 0
                self.mj.camera_vel[1] = 0
        elif chr(keycode) == "P":
            self.mj.watching = ((self.mj.watching or 0) - 1) % len(self.mj.meta)
        elif chr(keycode) == "N":
            self.mj.watching = ((self.mj.watching or 0) + 1) % len(self.mj.meta)
        elif chr(keycode) == "L":
            self.mj.watching = None
        elif chr(keycode) == "R":
            if self.mj.watching is not None:
                self.reload_code_cb(None, None, self.mj.watching)
        elif chr(keycode) == "S":
            self.mj.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] ^= 1
        else:
            # print(keycode)
            pass
        
    def run(self):
        print("GUI thread spawned")
        dpg.create_context()
        with dpg.item_handler_registry(tag="dog"):
            dpg.add_item_clicked_handler(0, callback=self.tracks_combo_clicked_cb)
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.scroll)
            dpg.add_mouse_release_handler(callback=self.release)
            dpg.add_mouse_drag_handler(callback=self.drag)
            dpg.add_key_release_handler(callback=self.release_key)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(1920, 1080, self.pixels, format=dpg.mvFormat_Float_rgb, tag="visualisation")
        with dpg.window(tag="Main", min_size=[400,400]):
            with dpg.group(tag="Main Group", horizontal=True):
                with dpg.group():
                    simulation_viewport_size = self.simulation_viewport_size()
                    dpg.add_image(
                        texture_tag="visualisation",
                        tag="simulation",
                        uv_max=[simulation_viewport_size[0] / 1920, simulation_viewport_size[1] / 1080],
                        width=simulation_viewport_size[0], height=simulation_viewport_size[1]
                    )
                dpg.add_spacer(width=10)
                with dpg.group(tag="Settings", width=200):
                    dpg.add_button(label="Reset Simulation", callback=self.reset)
                    dpg.add_button(label="Pause Simulation", callback=self.pause)
                    dpg.add_combo(tag="Tracks Combo",
                                  default_value=self.mj.map_metadata["name"],
                                  label="map",
                                  callback=self.select_map_callback)
                    dpg.add_separator()
                    with dpg.group(tag="Dashboard"): pass
                    dpg.add_separator()
                    dpg.add_slider_float(label="rangefinder intensity", default_value=0.1, max_value=0.3, callback=self.rangefinder)
        dpg.bind_item_handler_registry("Tracks Combo", "dog")
        dpg.create_viewport(title="Formula Trinity AI Grand Prix", width=self.window_size[0], height=self.window_size[1])
        dpg.set_viewport_resize_callback(self.viewport_resize_cb)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Main", True)
        dpg.start_dearpygui()

    def viewport_resize_cb(self, value, app):
        self.window_size = app[:2]
        simulation_viewport_size = self.simulation_viewport_size()
        dpg.configure_item(
            "simulation", 
            uv_max=[simulation_viewport_size[0] / 1920, simulation_viewport_size[1] / 1080],
            width=simulation_viewport_size[0],
            height=simulation_viewport_size[1]
        )
        self.mj.render()

    def tracks_combo_clicked_cb(self):
        items = [".".join(os.path.basename(f).split(".")[:-1])
                 for f in os.listdir(self.mj.template_dir)
                 if "svg" not in f and "png" in f]
        print("items are: ", items)
        dpg.configure_item("Tracks Combo", items=items)
        
    def select_map_callback(self, sender, value):
        self.mj.stage(value)

class Mujoco:
    # Use this if the rendered XML file must be changed e.g. the map is changed …
    def reload_shit(self, world=False):
        if world:
            # PROPHESY: You will spend days trying to fix an error caused due to
            # things being out of sync. This I have seen.
            map_metadata_path = os.path.join(self.rendered_dir, "chunks", "metadata.json")
            with open(map_metadata_path) as map_metadata_file:
                self.map_metadata = json.load(map_metadata_file)
            mjcf_metadata_path = os.path.join(self.rendered_dir, "car.json")
            with open(mjcf_metadata_path) as mjcf_metadata_file:
                self.mjcf_metadata = json.load(mjcf_metadata_file)
            self.model_path = os.path.join(self.rendered_dir, "car.xml")
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.model.vis.global_.offwidth = 1920
            self.model.vis.global_.offheight = 1080
            self.data = mujoco.MjData(self.model)
            mujoco.mj_kinematics(self.model, self.data)
            self.path = extract_path_from_svg(os.path.join(self.template_dir, f"{self.map_metadata['name']}-path.svg"))
            self.path[:, 0] =   self.path[:, 0] / self.map_metadata['width']  * self.map_metadata['chunk_width']
            self.path[:, 1] = - self.path[:, 1] / self.map_metadata['height'] * self.map_metadata['chunk_height']
        else:
            mujoco.mj_resetData(self.model, self.data)
        if world:
            metas = []
            for i, car in enumerate(self.mjcf_metadata["cars"]):
                if car["driver"].startswith("file://"):
                    path = car["driver"][7:-3].replace("/", ".")
                    print(f"Loading driver from path '{path}'")
                    driver = runtime_import(path).Driver()
                else:
                    print("Unsupported schema: supported (file://)")
                meta = Meta(
                    id          = i,
                    offset      = (i+5) * 2,
                    driver_path = path,
                    driver      = driver,
                    label       = car["name"],
                )
                metas.append(meta)
            self.meta = metas
            self.camera.lookat[:2] = self.path[0]
        for m in self.meta:
            m.reload_mujoco(self.data, rangefinders=self.mjcf_metadata["rangefinders"])
        # restart the renderer task
        # print("START")
        if world: self.render()
        # print("FINISH")
        self.position_vehicles(self.path)
    
    def stage(self, track):
        chunk(os.path.join(self.template_dir, f"{track}.png"), verbose=False, force=True)
        produce_mjcf(rangefinders=90, head=1)
        self.reload_shit(world=True)

    def __init__(self, mv, track):
        print("Running mujoco thread")
        self.reset_event = threading.Event()
        self.running_event = threading.Event()
        # self.running_event.set()
        self.mv = mv
        self.camera_vel = [0, 0]
        self.camera_friction = [0.01, 0.01]
        self.cinematic = False
        self.watching = None
        self.rendered_dir = "rendered"
        self.template_dir = "template"
        self.camera = mujoco.MjvCamera()

        self.render_exit     = threading.Event()
        self.render_finished = threading.Event()
        self.render_finished.set()
        
        self.stage(track)
        self.physics()

    def physics(self):
        threading.Thread(target=self.physics_thread).start()

    def render(self):
        if not self.render_finished.is_set():
            self.render_exit.set()
            self.render_finished.wait()
        threading.Thread(target=self.render_thread).start()

    def reload_code(self, index):
        self.meta[index].reload_code()

    def position_vehicles(self, path):
        # TODO: make sure that we can function without a
        # path. E.g. for testing maps or for when people make maps
        # without an SVG.
        
        for i, m in enumerate(self.meta):
            i = m.offset
            delta = path[i+1]-path[i]
            angle = math.atan2(delta[1], delta[0])
            quat = euler_to_quaternion([angle, 0, 0])
            m.joint.qpos[3:] = np.array(quat)
            m.joint.qpos[:2] = path[i]

    def physics_thread(self):
        global exit_event
        last = time.time()
        physics_steps = 0
        
        while True:
            if self.cinematic:
                self.camera_vel[0] *= (1 - self.camera_friction[0])
                self.camera_vel[1] *= (1 - self.camera_friction[1])
                self.camera.azimuth -= self.camera_vel[0]
                self.camera.elevation -= self.camera_vel[1]
            
            if self.watching is not None:
                self.camera.lookat[:] = self.data.body(f"car #{self.watching}").xpos[:]
            if self.reset_event.is_set():
                self.reload_shit()
                self.reset_event.clear()
                # print("RESET")
            if exit_event.is_set():
                break
            self.running_event.wait(timeout=0.25)
            if not self.running_event.is_set():
                mujoco.mj_forward(self.model, self.data)
                continue
            for index, meta in enumerate(self.meta):
                xpos = meta.joint.qpos[0:2]
                completion = (((self.path - xpos)**2).sum(1).argmin() - meta.offset) % 100
                if 45 <= completion <= 55:
                    meta.ready = True
                delta = completion - meta.completion
                if meta.ready and abs(delta) > 90:
                    lap_time = (physics_steps - meta.start) * 0.005
                    if delta > 0:
                        meta.laps -= 1
                        if len(meta.times) != 0:
                            meta.times.pop()
                    else:
                        meta.times.append(lap_time) # TODO: change this constant to always reflect physics timestep
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
        self.render_finished.clear()
        global exit_event
        last = time.time()
        width, height = self.mv.simulation_viewport_size()
        self.renderer = Renderer(self.model, width=width, height=height)
        buf = np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        max_fps = 5
        while True:
            # make it so that you can reset while paused …
            # self.running_event.wait()
            self.mv.inject_meta([meta.id for meta in self.meta])
            self.renderer.update_scene(self.data, self.camera)
            self.renderer.render(out=buf)
            self.mv.pixels[0:height, 0:width, :] = buf / 255.0
            now = time.time()
            if (now - last < 1/max_fps):
                time.sleep(1/max_fps - (now - last));
            # print (f"{1/(time.time()-last)} fps")
            last = time.time()
            if exit_event.is_set() or self.render_exit.is_set():
                self.renderer.close()
                self.render_finished.set()
                self.render_exit.clear()
                break

if __name__ == "__main__":
    try:
        ModelAndView(track="track").run()
    except KeyboardInterrupt:
        pass
    exit_event.set()
