# TODO: only render on user input when the thing is paused

# TODO: consider not using integer meta-keys and identifying cars
# using some means that is actually unique.

import os
import tempfile
import shutil
import math
import importlib
import queue
import json
import mujoco
import mujoco.viewer
import numpy as np
import time
from .colors import colors
import dearpygui.dearpygui as dpg
from .lobotomy import Driver as LobotomyDriver
from threading import Lock, Event, Thread
from .curve import extract_path_from_svg
from .chunk import chunk
from .map import produce_mjcf
from .vendor import Renderer

mujoco._mj_forward = mujoco.mj_forward
mujoco.mj_forward = lambda model, data: None
ignore = False

np.set_printoptions(precision=2, formatter={"float": lambda x: f"{x:8.2f}"})

def invert(d): return { v : k for k, v in d.items() }

def readable_keycode(keycode):
    try:
        return chr(keycode).encode("ascii").decode()
    except:
        return keycode

def ordinal(n):
    n = str(n)
    if n == '0' or len(n) > 1 and n[-2] == '1': e = 'th'
    elif (n[-1] == '1'): e = 'st'
    elif (n[-1] == '2'): e = 'nd'
    elif (n[-1] == '3'): e = 'rd'
    else:                e = 'th'
    return n + e

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

exit_event = Event()

class Real:
    def __init__(self, /, id):
        self.id = id
        self.completion_id = dpg.generate_uuid()
        self.laps_id = dpg.generate_uuid()
        self.position_id = dpg.generate_uuid()
        self.times_id = dpg.generate_uuid()
        self.pending_id = dpg.generate_uuid()
        self.finished_id = dpg.generate_uuid()
        self.finished_text_id = dpg.generate_uuid()

class Meta:
    def __init__(self, /, id, offset, driver, label, driver_path, data, rangefinders):
        # lifetime of race < lifetime of mujoco, ∴ mujoco is reloaded
        # more often than it needs to be.
        self.id = id
        self.offset = offset
        self.completion = 0 # current completion percentage
        self.good_start = True # if we enter into a lap backwards, set this to False
        self.start = 0 # physics time step we are currently at
        self.laps = 0 # number of laps completed
        self.times = [] # lap times
        self.driver = driver
        self.driver_path = driver_path
        self.label = label
        self.finished = False
        self.delta = 0
        self.distance_from_track = 0.0
        
        # lifetime of mujoco
        self.forward     = data.actuator(f"forward #{self.id}").id
        self.turn        = data.actuator(f"turn #{self.id}").id
        self.joint       = data.joint(f"car #{self.id}")
        self.sensors     = [data.sensor(f"rangefinder #{self.id}.#{j}").id for j in range(rangefinders)]

    def lap_completion(self):
        if self.good_start:
            return self.completion
        else:
            return -(100 - self.completion)
    
    def absolute_completion(self):
        return self.laps * 100 + self.lap_completion()

    def reload_code(self):
        self.driver = runtime_import(self.driver_path).Driver()
    
class ModelAndView:
    def __init__(self, track, width=1124, height=612):
        self.window_size = width, height
        self.pixels = np.zeros((1080, 1920, 3), dtype=np.float32)
        self.last = None
        self.mj = Mujoco(self, track=track)

        self.commands = {
            command.tag : command for command in [
                Command("pause", self.pause),
                Command("reset", self.reset),
                Command("focus_on_next_car", self.focus_on_next_car),
                Command("focus_on_previous_car", self.focus_on_previous_car),
                Command("reload_code",
                        lambda: self.reload_code_cb(None, None, self.mj.watching),
                        description="Reloads the code for the vehicle currently focused on"),
                Command("toggle_cinematic_camera", self.toggle_cinematic_camera),
                Command("liberate_camera", lambda: setattr(self.mj, "watching", None),
                        description="Stops focusing on any car"),
                Command("reload_code", lambda: self.reload_code_cb(None, None, self.mj.watching),
                        description="Reloads the code for the that is currently being watched"),
                Command("toggle_shadows", self.toggle_shadows),
                Command("rotate_camera_left", lambda: self.perturb_camera(-5, 0)),
                Command("rotate_camera_right", lambda: self.perturb_camera(5, 0)),
                Command("rotate_camera_up", lambda: self.perturb_camera(0, -5)),
                Command("rotate_camera_down", lambda: self.perturb_camera(0, 5))
            ]
        }
        
        self.keybindings = {
            " " : "pause",
            "E" : "reset",
            "R" : "reload_code",
            "N" : "focus_on_next_car",
            "P" : "focus_on_previous_car",
            "C" : "toggle_cinematic_camera",
            "L" : "liberate_camera",
            "S" : "toggle_shadows",
            263 : "rotate_camera_left",
            262 : "rotate_camera_right",
            265 : "rotate_camera_up",
            264 : "rotate_camera_down",
        }
        
    def toggle_shadows(self):
        """
        Toggles shadows. Not sure what to tell you
        """
        self.mj.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] ^= 1,

    def inject_options(self, options):
        for option in options.values():
            if not option.present:
                continue
            if type(option.value) is bool:
                dpg.add_checkbox(
                    parent="Options",
                    tag=option.dpg_tag,
                    label=option.label,
                    default_value=option.value,
                    callback=lambda sender, value, option: self.mj.option(option.tag, value),
                    user_data=option
                )
            elif type(option.value) is int:
                tag = dpg.add_input_int(
                    parent="Options",
                    label=option.label,
                    default_value=option.value,
                    callback=lambda sender, value, option: self.mj.option(option.tag, value),
                    user_data=option
                )
                if option.min_value:
                    dpg.configure_item(tag, min_value=option.min_value)
                if option.max_value:
                    dpg.configure_item(tag, max_value=option.max_value)
            elif type(option.value) is float:
                pass

    def simulation_viewport_size(self):
        return [max(self.window_size[0] - 400, 0),
                max(self.window_size[1] - 20, 0)]

    def set_inline_panel_visibility(self, visibility):
        if visibility:
            print("Making visible")
            dpg.configure_item("inline_panel", show=True)
            self.mj.kill_viewer_event.set()
        else:
            print("Making invisible")
            dpg.configure_item("inline_panel", show=False)
            self.mj.launch_viewer_event.set()
    
    def inject_meta(self, meta):
        positions = { m  : i for i, m in enumerate(reversed(sorted(meta, key=lambda i: self.mj.meta[i].absolute_completion()))) }
        meta = set(meta) # payload being injected
        for id in dpg.get_item_children("dashboard", 1):
            real = dpg.get_item_configuration(id)['user_data']
            m = self.mj.meta[real.id]
            if m.id not in meta:
                dpg.delete_item(id)
            else:
                if real.id in meta:
                    meta.remove(real.id)
                dpg.configure_item(real.pending_id, show=not m.finished)
                dpg.configure_item(real.finished_id, show=m.finished)
                dpg.set_value(real.times_id, "[" + ", ".join([f"{t:.2f}" for t in m.times]) + "]")
                position = positions[real.id]
                if position == 0:   color = colors["gold"]
                elif position == 1: color = colors["silver"]
                elif position == 2: color = colors["brown"]
                else:               color = None
                dpg.set_value(real.position_id, ordinal(position + 1))
                dpg.configure_item(real.position_id, color=color)
                if not m.finished:
                    dpg.set_value(real.completion_id, str(m.lap_completion()) + "%")
                    dpg.set_value(real.laps_id, str(m.laps))
                else:
                    dpg.set_value(real.finished_text_id, f"Car finished {m.laps} laps in {sum(m.times):.2f} seconds! ({ordinal(self.mj.winners[m.id])} place)")
        for m in meta:
            real = Real(m)
            m = self.mj.meta[m]
            with dpg.tree_node(user_data=real, parent="dashboard", label=f"Car #{m.id} - {m.label}", default_open=True):
                with dpg.group(tag=real.pending_id):
                     with dpg.group(horizontal=True):
                         dpg.add_text(f"Completion:")
                         dpg.add_text("0%", tag=real.completion_id)
                     with dpg.group(horizontal=True):
                         dpg.add_text(f"Laps:")
                         dpg.add_text("0", tag=real.laps_id)
                with dpg.group(tag=real.finished_id):
                    dpg.add_text("", tag=real.finished_text_id)
                with dpg.group(horizontal=True):
                    dpg.add_text("Position:")
                    dpg.add_text("", tag=real.position_id)
                with dpg.group(horizontal=True):
                    dpg.add_text("Lap Times:")
                    dpg.add_text("0", tag=real.times_id, wrap=300)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reload Code", user_data=real.id, callback=self.reload_code_cb)
                    dpg.add_button(label="Info", user_data=real.id)

    def reload_code_cb(self, sender, value, user_data):
        if dpg.does_item_exist("reload code modal"):
            dpg.delete_item("reload code modal")
        car_index = user_data
        if car_index is None:
            return
        try:
            self.mj.reload_code(car_index)
        except Exception as e:
            with dpg.window(tag="reload code modal", modal=True, show=True) as window:
                dpg.add_text(str(e), color=[255, 200, 200])
                dpg.add_button(label="Try Again", callback=self.reload_code_cb, user_data=car_index)
    
    def pause(self):
        """
        Resets the simulation
        """
        if self.mj.running_event.is_set():
            self.mj.running_event.clear()
        else:
            self.mj.running_event.set()

    def reset(self):
        """
        Pauses the simulation
        """
        self.mj.reset_event.set()

    def scroll(self, sender, value):
        state = dpg.get_item_state('simulation')
        [x, y] = dpg.get_mouse_pos(local=False)
        bounded = \
            state['rect_min'][0] <= x < state['rect_max'][0] \
            and state['rect_min'][1] <= y < state['rect_max'][1]
        if not bounded or self.mj.viewer is not None:
            return
        self.mj.camera.distance -= value / 20
        
    def release(self):
        self.last = None

    def drag(self, sender, value):
        state = dpg.get_item_state('simulation')
        [x, y] = dpg.get_mouse_pos(local=False)
        bounded = \
            state['rect_min'][0] <= x < state['rect_max'][0] \
            and state['rect_min'][1] <= y < state['rect_max'][1]
        if not bounded or self.mj.viewer is not None:
            return
        delta = [dx, dy] = dpg.get_mouse_drag_delta()
        if self.last is not None:
            dx -= self.last[0]
            dy -= self.last[1]
        self.last = delta
        self.perturb_camera(dx, dy)

    def perturb_camera(self, dx, dy):
        if not self.mj.option("cinematic_camera"):
            self.mj.camera.azimuth   += dx
            self.mj.camera.elevation += dy
        else:
            self.mj.camera_vel[0] += dx / 100
            self.mj.camera_vel[1] += dy / 100
        
    def rangefinder(self, sender, value):
        self.mj.model.vis.rgba.rangefinder[3] = value

    def focus_on_next_car(self):
        """
        Locks the camera to the next car in the sequence. Cycles at the end
        """
        self.mj.watching = ((self.mj.watching or 0) - 1) % len(self.mj.meta)

    def focus_on_previous_car(self):
        """
        Locks the camera to the previous car in the sequence. Cycles at the end
        """
        self.mj.watching = ((self.mj.watching or 0) - 1) % len(self.mj.meta)

    def toggle_cinematic_camera(self):
        """
        Toggle the cinemaic camera value. If cinematic mode is turned off as a result,
        the camera's velocity is set to zero.
        """
        self.mj.option("cinematic_camera", not self.mj.option("cinematic_camera"))
        if not self.mj.option("cinematic_camera"):
            self.mj.camera_vel[0] = 0
            self.mj.camera_vel[1] = 0

    def release_key(self, sender, keycode):
        command = self.keybindings.get(keycode) or self.keybindings.get(chr(keycode))
        if command is not None:
            command = self.commands[command]
            print(command.tag)
            command.callback()
        else:
            print(f"Unbound keycode {keycode}")
        
    def run(self):
        print("GUI thread spawned")
        dpg.create_context()
        with dpg.item_handler_registry(tag="map_combo_handler_registry"):
            dpg.add_item_clicked_handler(0, callback=self.tracks_combo_clicked_cb)
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.scroll)
            dpg.add_mouse_release_handler(callback=self.release)
            dpg.add_mouse_drag_handler(callback=self.drag)
            dpg.add_key_release_handler(callback=self.release_key)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(1920, 1080, self.pixels, format=dpg.mvFormat_Float_rgb, tag="visualisation")
        with dpg.window(tag="main", min_size=[400,400]):
            with dpg.group(tag="Main Group", horizontal=True):
                with dpg.group(tag="inline_panel", show=True):
                    simulation_viewport_size = self.simulation_viewport_size()
                    dpg.add_image(
                        texture_tag="visualisation",
                        tag="simulation",
                        uv_max=[simulation_viewport_size[0] / 1920, simulation_viewport_size[1] / 1080],
                        width=simulation_viewport_size[0], height=simulation_viewport_size[1],
                    )
                    dpg.add_spacer(width=10)
                with dpg.child_window(tag="settings", width=375):
                    dpg.add_button(label="Reset Simulation", callback=self.reset)
                    dpg.add_button(label="Pause Simulation", callback=self.pause)
                    dpg.add_checkbox(label="External Viewer (faster)",
                                     callback=lambda sender, value: self.set_inline_panel_visibility(not value))
                    dpg.add_combo(tag="Tracks Combo",
                                  default_value=self.mj.map_metadata["name"],
                                  label="map",
                                  callback=self.select_map_callback)
                    dpg.add_separator()
                    with dpg.collapsing_header(label="Vehicle Overview", tag="dashboard", default_open=True):
                        pass
                    dpg.add_separator()
                    with dpg.collapsing_header(label="Options", tag="Options", default_open=True):
                        dpg.add_button(label="Keybindings Info", callback=self.show_keybindings)
                        dpg.add_separator()
                        pass
                    
                    # change when there are more options
                    # dpg.add_separator()
                    # dpg.add_slider_float(label="rangefinder intensity", default_value=0.1, max_value=0.3, callback=self.rangefinder)
        dpg.bind_item_handler_registry("Tracks Combo", "map_combo_handler_registry")
        dpg.create_viewport(title="Formula Trinity AI Grand Prix", width=self.window_size[0], height=self.window_size[1])
        dpg.set_viewport_resize_callback(self.viewport_resize_cb)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

        last = time.time()
        self.inject_options(self.mj.options)
        while dpg.is_dearpygui_running():
            now = time.time()
            if (now - last < 1/self.mj.option("max_fps")):
                time.sleep(1/self.mj.option("max_fps") - (now - last));
            # print (f"{1/(time.time()-last)} fps")
            last = time.time()
            dpg.render_dearpygui_frame()
            self.inject_meta([meta.id for meta in self.mj.meta])
        dpg.destroy_context()
    
    def succ_keys(self, sender, keycode, command):
        readable = readable_keycode(keycode)
        conflict = self.keybindings.get(keycode) or self.keybindings.get(readable)
        if conflict is None or conflict == command.tag:
            with dpg.handler_registry():
                dpg.add_key_release_handler(callback=self.release_key)
            dpg.configure_item("keybindings select", show=False)
            dpg.configure_item("keybindings dashboard", show=True)
            # just restore original values if ESC was pressed
            if keycode == 256:
                existing_keycode = invert(self.keybindings).get(command.tag)
                if existing_keycode is not None:
                    del self.keybindings[existing_keycode]
                dpg.configure_item(f"keybinding for {command.tag}", label=f"N/A")
            else:
                # FIXME: store in unreadable form
                self.keybindings[keycode] = command.tag
                dpg.configure_item(f"keybinding for {command.tag}", label=f"`{readable}`")
        else:
            command = self.commands[conflict]
            dpg.set_value("keybindings message", f"`{readable}` would clash with command `{command.tag}`")

    def supplant(self, sender, value, command):
        dpg.configure_item("keybindings dashboard", show=False)
        dpg.configure_item("keybindings select", show=True)
        dpg.set_value("keybindings target", f"Setting keybinding for `{command.tag}`")
        
        with dpg.handler_registry():
            dpg.add_key_release_handler(callback=self.succ_keys, user_data=command)

    def show_keybindings(self):
        inverted_keybindings_index = invert(self.keybindings)

        # FIXME: Exiting the keybindings screen early disables all
        # further key presses as the callback isn't restored
        # correctly.
        if dpg.does_item_exist("keybindings modal"):
            dpg.delete_item("keybindings modal")
        with dpg.window(tag="keybindings modal", modal=True):
            with dpg.group(tag="keybindings select", show=False):
                dpg.add_text(tag="keybindings target")
                dpg.add_text("waiting for user input ...", tag="keybindings message", color=colors["silver"])
                dpg.add_text("(press escape to unbind)")
            with dpg.group(tag="keybindings dashboard"):
                for command in self.commands.values():
                    with dpg.group(horizontal=True):
                        dpg.add_text(command.label + ":")
                        key = inverted_keybindings_index.get(command.tag)
                        if key is None: label = "N/A"
                        else:           label = f"`{key}`"
                        dpg.add_button(tag=f"keybinding for {command.tag}",
                                       label=label,
                                       callback=self.supplant,
                                       user_data=command)
                    if command.description is not None and command.description != "":
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=20)
                            with dpg.group():
                                dpg.add_text(
                                    command.description.strip().replace("\n", " "),
                                    wrap=350,
                                    color=colors["silver"]
                                )

    def viewport_resize_cb(self, value, app):
        self.window_size = app[:2]
        simulation_viewport_size = self.simulation_viewport_size()
        dpg.configure_item("settings", height=simulation_viewport_size[1])
        if self.mj.viewer is None:
            dpg.configure_item("settings", width=375)
        else:
            dpg.configure_item("settings", width=self.window_size[0] - 20)
        dpg.configure_item(
            "simulation", 
            uv_max=[simulation_viewport_size[0] / 1920, simulation_viewport_size[1] / 1080],
            width=simulation_viewport_size[0],
            height=simulation_viewport_size[1]
        )
        # print("START")
        self.mj.render()
        # print("FINISH")

    def tracks_combo_clicked_cb(self):
        items = [".".join(os.path.basename(f).split(".")[:-1])
                 for f in os.listdir(self.mj.template_dir)
                 if "svg" not in f and "png" in f]
        print("items are: ", items)
        dpg.configure_item("Tracks Combo", items=items)
        
    def select_map_callback(self, sender, value):
        self.mj.stage(value)

# A simple JSON option that can be persisted in the file system
class Option:
    def __init__(self, tag, default, _type=None, description="", data=None, label=None, persist=True, min_value=None, max_value=None, present=True):
        self.tag = tag
        self.dpg_tag = f"__option__::{tag}"
        self.label = label or tag
        self.persist = persist
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.value = default
        self.present = present
        self.type = _type or type(default)
        if self.tag in data:
            their = type(data[self.tag])
            if data[self.tag] is not None and self.type is not their:
                print (f"{tag} was saved as {their}, using default instead ({self.type})")
                self.value = default
            else:
                self.value = data[self.tag]

class Command:
    def __init__(self, tag, callback, label=None, description=None):
        self.tag = tag
        self.label = label or tag
        self.callback = callback
        self.description = description or getattr(callback, "__doc__", None)

class Mujoco:
    def __init__(self, mv, track):
        print("Running mujoco thread")
        self.reset_event = Event()
        self.running_event = Event()
        self.launch_viewer_event = Event()
        self.kill_viewer_event = Event()
        self.mv = mv
        self.camera_vel = [0, 0]
        self.camera_friction = [0.01, 0.01]
        self.watching = 0
        self.rendered_dir = "rendered"
        self.template_dir = "template"
        self.camera = mujoco.MjvCamera()
        self.options = {}
        try:
            with open("aigp_settings.json") as data_file:
                data = json.load(data_file)
        except FileNotFoundError as e:
            data = {}
            print("Not loading settings from file")
        
        self.declare("lap_target", 10, data=data, label="Lap Target")
        self.declare("max_fps", 30, data=data, label="Max FPS")
        self.declare("cinematic_camera", False, data=data, label="Cinematic Camera")
        self.declare("pause_on_reload", True, data=data, label="Pause on Reload")
        self.declare("save_on_exit", True, data=data, label="Save on Exit", present=False)

        self.meta = []
        self.shadows = {}
        self.steps = 0

        self.viewer = None
        self.kill_inline_render_event = Event()
        self.render_finished = Event()
        self.render_finished.set()
        
        self.stage(track)
        self.physics()
        
    def persist(self):
        if self.option("save_on_exit"):
            options = {
                option.tag : option.value
                for option in self.options.values()
                if option.persist
            }
            path = tempfile.mktemp()
            with open(path, "w") as file:
                json.dump(options, file)
            shutil.copy(path, "aigp_settings.json")

    def declare(self, tag, default, **kwargs):
        self.options[tag] = Option(tag, default, **kwargs)

    def option(self, tag, value=None):
        if value is not None:
            # print(f"`{tag}` = `{value}`")
            self.options[tag].value = value
        return self.options[tag].value
    
    def reload(self):
        if self.option("pause_on_reload"):
            self.running_event.clear()
        mujoco.mj_resetData(self.model, self.data)
        for m in self.meta:
            self.unshadow(m.id)
        metas = []
        for i, car in enumerate(self.mjcf_metadata["cars"]):
            if car["driver"].startswith("file://"):
                path = car["driver"][7:-3].replace("/", ".")
                print(f"Loading driver from python module path '{path}'")
                driver = runtime_import(path).Driver()
            else:
                print("Unsupported schema: supported (file://)")
            meta = Meta(
                id           = i,
                offset       = (i+5) * 2,
                driver_path  = path,
                driver       = driver,
                label        = car["name"],
                data         = self.data,
                rangefinders = self.mjcf_metadata["rangefinders"]
            )
            metas.append(meta)
        self.meta = metas
        self.shadows = {}
        self.steps = 0
        self.winners = {}
        self.camera.lookat[:2] = self.path[0]
        self.position_vehicles(self.path)
    
    def stage(self, track):
        chunk(os.path.join(self.template_dir, f"{track}.png"), verbose=False, force=True)
        produce_mjcf(rangefinders=90, head=4)
        map_metadata_path = os.path.join(self.rendered_dir, "chunks", "metadata.json")
        with open(map_metadata_path) as map_metadata_file:
            self.map_metadata = json.load(map_metadata_file)
        mjcf_metadata_path = os.path.join(self.rendered_dir, "car.json")
        with open(mjcf_metadata_path) as mjcf_metadata_file:
            self.mjcf_metadata = json.load(mjcf_metadata_file)
        self.model_path = os.path.join(self.rendered_dir, "car.xml")
        self.original_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        self.path = extract_path_from_svg(os.path.join(self.template_dir, f"{self.map_metadata['name']}-path.svg"))
        self.path[:, 0] =   self.path[:, 0] / self.map_metadata['width']  * self.map_metadata['chunk_width']
        self.path[:, 1] = - self.path[:, 1] / self.map_metadata['height'] * self.map_metadata['chunk_height']
        self.render()
        self.reload()

    def physics(self):
        Thread(target=self.physics_thread).start()

    def render(self):
        if self.viewer is None:
            if not self.render_finished.is_set():
                self.kill_inline_render_event.set()
                self.render_finished.wait()
            Thread(target=self.inline_render_thread).start()
        else:
            self.launch_viewer_event.set()

    def reload_code(self, index):
        self.meta[index].reload_code()

    def position_vehicles(self, path):
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
        while True:
            if self.option("cinematic_camera"):
                self.camera_vel[0] *= (1 - self.camera_friction[0])
                self.camera_vel[1] *= (1 - self.camera_friction[1])
                self.camera.azimuth += self.camera_vel[0]
                self.camera.elevation += self.camera_vel[1]
            if self.launch_viewer_event.is_set():
                self.kill_inline_render_event.set()
                if self.viewer is not None:
                    self.viewer.close()
                self.viewer = "John the Baptist"
                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback = lambda keycode: self.mv.release_key(None, keycode)
                )
                self.launch_viewer_event.clear()
            if self.kill_viewer_event.is_set():
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                self.kill_viewer_event.clear()
            if self.watching is not None:
                self.camera.lookat[:] = self.data.body(f"car #{self.watching}").xpos[:]
            if self.reset_event.is_set():
                self.reload()
                self.reset_event.clear()
            if exit_event.is_set():
                self.persist()
                print("Persisted!")
                if self.viewer is not None:
                    self.viewer.close()
                break
            
            # synchronise with the viewer before we short circuit
            if self.viewer is not None:
                self.viewer.cam.lookat = self.camera.lookat
                
                # debatable whether these should be shared
                # self.viewer.cam.azimuth = self.camera.azimuth
                # self.viewer.cam.elevation = self.camera.elevation
                
                with self.viewer.lock():
                    scene = self.viewer.user_scn
                    scene.ngeom = 0
                    for shadows in self.shadows.values():
                         for shadow in shadows:
                             mujoco.mjv_initGeom(scene.geoms[scene.ngeom], **shadow)
                             scene.geoms[scene.ngeom].dataid = 0
                             scene.ngeom += 1
                    self.viewer.sync()
            
            self.running_event.wait(timeout=1/self.option("max_fps"))
            if not self.running_event.is_set():
                mujoco._mj_forward(self.model, self.data)
                continue
            for index, meta in enumerate(self.meta):
                xpos = meta.joint.qpos[0:2]
                distances = ((self.path - xpos)**2).sum(1)
                closest = distances.argmin()
                meta.distance_from_track = distances[closest]
                meta.off_track = meta.distance_from_track > 0.5
                if meta.off_track:
                    print(f"Car {index} is off track")
                    continue
                        
                completion = (closest - meta.offset) % 100
                
                delta = completion - meta.completion
                meta.delta = (completion - meta.completion + 50) % 100 - 50
                
                if abs(delta) > 90:
                    lap_time = (self.steps - meta.start) * self.model.opt.timestep
                    if meta.delta < 0:
                        meta.good_start = False
                        meta.laps -= 1
                        if len(meta.times) != 0:
                            meta.times.pop()
                    elif meta.delta > 0:
                        if meta.good_start:
                            # dont add a new lap OR start counting
                            # time for the next lap if we went
                            # backwards, we are still in the current
                            # lap
                            meta.times.append(lap_time)
                            meta.start = self.steps
                        meta.laps += 1
                        meta.good_start = True
                if meta.laps >= self.option("lap_target"):
                    if meta.id not in self.winners:
                        self.winners[meta.id] = len (self.winners) + 1
                    meta.finished = True
                    self.shadow(meta.id)
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
            self.steps += 1
            
            now = time.time()
            fps = 1 / (now - last)
            last = time.time()
            # print(f"{fps} fps")

    # send car to the shadow realm
    def shadow(self, i):
        if i in self.shadows:
            return self.shadows[i]
        meta = self.meta[i]
        # lobotomise car first
        meta.driver = LobotomyDriver()
        meta.driver_path = "ft_grandprix.lobotomy"
        shadows = []
        for j in range(self.mjcf_metadata["rangefinders"]):
            self.model.sensor(f"rangefinder #{i}.#{j}").type = mujoco.mjtSensor.mjSENS_USER
        transparent_material_id  = self.model.mat("transparent").id
        shadow_alpha = 0.1
        for mat in [f"car #{i} primary", f"car #{i} secondary", f"car #{i} body", f"car #{i} wheel", f"car #{i} icon"]:
            self.model.mat(mat).rgba[3] = shadow_alpha
        for geom in [f"chasis #{i}", f"car #{i} lidar", f"front wheel #{i}", f"left wheel #{i}", f"right wheel #{i}"]:
            m, d = self.model.geom(geom), self.data.geom(geom)
            r = [*self.model.mat(m.matid.item()).rgba[:3], shadow_alpha]
            shadow = dict(type=m.type.item(), size=m.size, pos=d.xpos, rgba=r, mat=d.xmat)
            # turn off collisions (only makes sense wrt car.em.xml condim and
            # conaffinity values)
            m.conaffinity = 0
            m.contype = 1 << 1
            m.matid = transparent_material_id
            shadows.append(shadow)
        self.shadows[meta.id] = shadows
        return self.shadows[meta.id]

    def unshadow(self, i):
        if i not in self.shadows:
            return
        for j in range(90):
            self.model.sensor(f"rangefinder #{i}.#{j}").type = mujoco.mjtSensor.mjSENS_RANGEFINDER
        for mat in [f"car #{i} primary", f"car #{i} secondary", f"car #{i} body", f"car #{i} wheel", f"car #{i} icon"]:
            self.model.mat(mat).rgba[3] = 1.0
        for geom in [f"chasis #{i}", f"car #{i} lidar", f"front wheel #{i}", f"left wheel #{i}", f"right wheel #{i}"]:
            m = self.model.geom(geom)
            m.conaffinity = 1
            m.contype = 1
            m.matid = self.original_model.geom(geom).matid
        del self.shadows[i]

    def inline_render_thread(self):
        global exit_event
        self.render_finished.clear()
        last = time.time()
        width, height = self.mv.simulation_viewport_size()
        self.renderer = Renderer(self.model, width=width, height=height)
        scene = self.renderer.scene
        buf = np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        while True:
            self.renderer.update_scene(self.data, self.camera)
            for shadows in self.shadows.values():
                for shadow in shadows:
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], **shadow)
                    self.renderer.scene.geoms[scene.ngeom].dataid = 0
                    scene.ngeom += 1
            self.renderer.render(out=buf)
            self.mv.pixels[0:height, 0:width, :] = buf / 255.0
            now = time.time()
            if (now - last < 1/self.option("max_fps")):
                time.sleep(1/self.option("max_fps") - (now - last));
            # print (f"{1/(time.time()-last)} fps")
            last = time.time()
            if exit_event.is_set() or self.kill_inline_render_event.is_set():
                self.renderer.close()
                self.render_finished.set()
                self.kill_inline_render_event.clear()
                break

if __name__ == "__main__":
    try:
        ModelAndView(track="track").run()
    except KeyboardInterrupt:
        pass
    exit_event.set()
