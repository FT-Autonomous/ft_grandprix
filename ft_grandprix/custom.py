from sys import platform
import os
import re
import inspect
from PIL import Image
import tempfile
import shutil
import math
import importlib
import queue
import json
from dataclasses import dataclass, field
from .vehicle import VehicleStateSnapshot
import mujoco
import mujoco.viewer
import numpy as np
import time
from .colors import colors, resolve_color
import dearpygui.dearpygui as dpg
from .lobotomy import Driver as LobotomyDriver
from threading import Lock, Event, Thread
from .curve import extract_path_from_svg
from .chunk import chunk
from .map import produce_mjcf
from .vendor import Renderer
from .raycast import fakelidar
from .bracket import compute_driver_files
import tracemalloc
import linecache

def tag():
    return field(default_factory=dpg.generate_uuid)

# monkey patch so that the viewer doesn't do its own forwarding
mujoco._mj_forward = mujoco.mj_forward
mujoco.mj_forward = lambda model, data: None

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

def quaternion_to_euler(w, x, y, z):
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
    
    return [yaw_z, pitch_y, roll_x]

def quaternion_to_angle(w, x, y, z):
    return quaternion_to_euler(w, x, y, z)[0]

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    return [qw, qx, qy, qz]

exit_event = Event()

class VehicleState:
    def __init__(self, /, id, offset, driver, label, driver_path, data, rangefinders):
        # lifetime of race < lifetime of mujoco, ∴ mujoco is reloaded
        # more often than it needs to be.
        self.id = id
        self.offset = offset
        self.completion = 0 # current completion percentage
        self.good_start = True # if we enter into a lap backwards, set this to False
        self.driver = driver
        self.driver_path = driver_path
        self.finished = False
        self.delta = 0
        self.v2 = len(inspect.signature(self.driver.process_lidar).parameters) >= 2

        self.speed = 0.0
        """the last speed command"""

        self.steering_angle = 0.0
        """the last steering angle command"""

        self.distance_from_track = 0.0
        """distance from the centerline"""

        self.label = label
        """the name of this driver"""
        
        self.start = 0
        """physics time step we are currently at"""

        self.laps = 0
        """number of laps completed"""

        self.times = []
        """lap times so far"""

        # lifetime of mujoco
        self.forward     = data.actuator(f"forward #{self.id}").id
        self.turn        = data.actuator(f"turn #{self.id}").id
        self.joint       = data.joint(f"car #{self.id}")
        self.sensors     = [data.sensor(f"rangefinder #{self.id}.#{j}").id for j in range(rangefinders)]

    def lap_completion(self):
        """
        Gets our completion around the track. Returns a negative number if
        we are going backwards.
        """
        if self.good_start:
            return self.completion
        else:
            return -(100 - self.completion)
    
    def absolute_completion(self):
        return self.laps * 100 + self.lap_completion()

    def reload_code(self):
        self.driver = runtime_import(self.driver_path).Driver()
        self.v2 = len(inspect.signature(self.driver.process_lidar).parameters) >= 2

    def snapshot(self, time=0):
        yaw, pitch, roll = quaternion_to_euler(*self.joint.qpos[3:])
        return VehicleStateSnapshot(
            laps = self.laps,
            velocity=self.joint.qvel[:3],
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            lap_completion=self.lap_completion(),
            absolute_completion=self.absolute_completion(),
            time=time
        )
    
class ModelAndView:
    def __init__(self, track, width=1124, height=612):
        print("Running on", platform)
        self.viewport_resize_event = Event()
        self.window_size = width, height
        if platform == "darwin":
            self.pixels = np.zeros((1080, 1920, 4), dtype=np.float32)
            self.pixels[..., 3] = 1.0
        else:
            self.pixels = np.zeros((1080, 1920, 3), dtype=np.float32)
        self.last = None
        self.speed = 0.0
        self.steering_angle = 0.0
        self.mj = Mujoco(self, track=track)
        self.mj.run()

        self.commands = {
            command.tag : command for command in [
                Command("pause", self.pause),
                Command("reset", self.reset),
                Command("hard_reset", self.hard_reset),
                Command("zoom_in", lambda: self.scroll_cb(None, 10)),
                Command("zoom_out", lambda: self.scroll_cb(None, -10)),
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
                Command("rotate_camera_left", lambda: self.mj.perturb_camera(-5, 0)),
                Command("rotate_camera_right", lambda: self.mj.perturb_camera(5, 0)),
                Command("rotate_camera_up", lambda: self.mj.perturb_camera(0, -5)),
                Command("rotate_camera_down", lambda: self.mj.perturb_camera(0, 5)),
                Command("show_cars_modal", self.show_cars_modal),
                Command("show_keybindings_modal", self.show_keybindings_modal),
                Command("sync_options", self.mj.sync_options),
                Command("position_vehicles", self.mj.position_vehicles) # possibly not thread safe
            ]
        }
        
        self.keybindings = {
            " " : "pause",
            "E" : "reset",
            "R" : "reload_code",
            "-" : "zoom_out",
            "+" : "zoom_in",
            "N" : "focus_on_next_car",
            "P" : "focus_on_previous_car",
            "C" : "toggle_cinematic_camera",
            "L" : "liberate_camera",
            "H" : "toggle_shadows",
            263 : "rotate_camera_left",
            262 : "rotate_camera_right",
            265 : "rotate_camera_up",
            264 : "rotate_camera_down",
        }
        
    def toggle_shadows(self):
        """
        Toggles shadows. Not sure what to tell you
        """
        self.mj.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] ^= 1

    def inject_options(self, options):
        for option in options.values():
            if dpg.does_item_exist(option.dpg_tag):
                if not self.mj.option("debug_mode") and not option.present:
                    for tag in [option.dpg_tag, option.dpg_description_tag]:
                        try:
                            dpg.delete_item(tag)
                        except:
                            pass
                elif option.type is not list:
                    dpg.set_value(option.dpg_tag, option.value if option.value is not None else option.type())
            elif not self.mj.option("debug_mode") and not option.present:
                continue
            else:
                default_attributes = dict(
                    tag=option.dpg_tag,
                    label=option.label,
                    default_value=option.value if option.value is not None else option.type(),
                    callback=lambda sender, value, option: self.mj.option(option.tag, value),
                    user_data=option,
                    width=-200
                )
                with dpg.group(parent="Options"):
                    if option.type is bool:
                        del default_attributes["width"]
                        dpg.add_checkbox(**default_attributes)
                    elif option.type is int:
                        dpg.add_input_int(**default_attributes)
                        if option.min_value:
                            dpg.configure_item(option.dpg_tag, min_value=option.min_value)
                        if option.max_value:
                            dpg.configure_item(option.dpg_tag, max_value=option.max_value)
                    elif option.type is float:
                        dpg.add_input_float(**default_attributes)
                        if option.min_value:
                            dpg.configure_item(option.dpg_tag, min_value=option.min_value)
                        if option.max_value:
                            dpg.configure_item(option.dpg_tag, max_value=option.max_value)
                    elif option.type is str:
                        if default_attributes["default_value"] is None:
                            default_attributes["default_value"] = ""
                        dpg.add_input_text(**default_attributes)
                    elif option.type is list: # dirty hack
                        dpg.add_color_edit(**default_attributes)
                    if option.description is not None:
                        description = re.sub(r"\s+", " ", option.description.strip())
                        with dpg.group(tag=option.dpg_description_tag, horizontal=True):
                            dpg.add_spacer(width=20)
                            dpg.add_text(description, wrap=300, color=colors["silver"])
                            
    def simulation_viewport_size(self):
        return [min(self.mj.model.vis.global_.offwidth, max(self.window_size[0] - 400, 0)),
                min(self.mj.model.vis.global_.offheight, max(self.window_size[1] - 20, 0))]

    def set_inline_panel_visibility(self, visibility):
        if visibility:
            dpg.configure_item("inline_panel", show=True)
            self.mj.kill_viewer_event.set()
        else:
            dpg.configure_item("inline_panel", show=False)
            self.mj.launch_viewer_event.set()
    
    def inject_vehicle_state(self, vehicle_state):
        @dataclass
        class Tags:
            completion_id    : int = tag()
            laps_id          : int = tag()
            position_id      : int = tag()
            times_id         : int = tag()
            pending_id       : int = tag()
            finished_id      : int = tag()
            finished_text_id : int = tag()
            reload_id        : int = tag()
            watch_id         : int = tag()
            tag              : int = tag()
        mj_vehicle_state = self.mj.vehicle_states # hack to make sure that this is atomic
        tags = [dpg.get_item_configuration(tag)["user_data"] for tag in dpg.get_item_children("dashboard", 1)]
        delta = len(mj_vehicle_state) - len(tags)
        if delta < 0:
            # delete the first few that we dont need
            for i in range(-delta): dpg.delete_item(tags.pop().tag)
        elif delta > 0:
            # add the ones that we do not have
            for i in range(delta):
                tags.append(Tags())
                tag = tags[-1]
                with dpg.tree_node(tag=tag.tag, user_data=tag, parent="dashboard", default_open=True):
                    with dpg.group(tag=tag.pending_id):
                         with dpg.group(horizontal=True):
                             dpg.add_text(f"Completion:")
                             dpg.add_text("0%", tag=tag.completion_id)
                         with dpg.group(horizontal=True):
                             dpg.add_text(f"Laps:")
                             dpg.add_text("0", tag=tag.laps_id)
                    with dpg.group(tag=tag.finished_id):
                        dpg.add_text("", tag=tag.finished_text_id)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Position:")
                        dpg.add_text("", tag=tag.position_id)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Lap Times:")
                        dpg.add_text("0", tag=tag.times_id, wrap=300)
                    with dpg.group(horizontal=True):
                        dpg.add_button(tag=tag.reload_id, label="Reload Code", user_data=i, callback=self.reload_code_cb)
                        dpg.add_button(tag=tag.watch_id, label="Watch", user_data=i, callback=self.watch_cb)
        positions = { m  : i for i, m in enumerate(reversed(sorted(vehicle_state, key=lambda i: mj_vehicle_state[i].absolute_completion()))) }
        if self.mj.option("sort_vehicle_list"):
            vehicle_state = sorted(vehicle_state, key=lambda v: positions[v])
        for tag, m_id in zip(tags, vehicle_state):
            m = mj_vehicle_state[m_id]
            dpg.configure_item(tag.tag, label=f"Car #{m.id} - {m.label}")
            dpg.configure_item(tag.pending_id, show=not m.finished)
            dpg.configure_item(tag.finished_id, show=m.finished)
            dpg.configure_item(tag.reload_id, user_data=m_id)
            dpg.configure_item(tag.watch_id, user_data=m_id)
            dpg.set_value(tag.times_id, "[" + ", ".join([f"{t:.2f}" for t in m.times]) + "]")
            position = positions[m.id]
            if position == 0:
                color = colors["gold"]
            elif position == 1:
                color = colors["silver"]
            elif position == 2:
                color = colors["brown"]
            else:
                color = None
            dpg.set_value(tag.position_id, ordinal(position + 1))
            dpg.configure_item(tag.position_id, color=color)
            if not m.finished:
                dpg.set_value(tag.completion_id, str(m.lap_completion()) + "%")
                dpg.set_value(tag.laps_id, str(m.laps))
            else:
                dpg.set_value(tag.finished_text_id, f"Car finished {m.laps} laps in {sum(m.times):.2f} seconds!")

    def watch_cb(self, sender, value, car_index):
        self.mj.watching = car_index

    def reload_code_cb(self, sender, value, car_index):
        if car_index is None:
            return
        try:
            self.mj.reload_code(car_index)
        except Exception as e:
            if dpg.does_item_exist("reload code modal"):
                dpg.delete_item("reload code modal")
            with dpg.window(tag="reload code modal", modal=True, show=True) as window:
                dpg.add_text(str(e), color=colors["error"])
                dpg.add_button(label="Try Again", callback=self.reload_code_cb, user_data=car_index)

    def scroll_cb(self, sender, value):
        if self.modal_visible(): return
        state = dpg.get_item_state('simulation')
        [x, y] = dpg.get_mouse_pos(local=False)
        bounded = \
            state['rect_min'][0] <= x < state['rect_max'][0] \
            and state['rect_min'][1] <= y < state['rect_max'][1]
        if not bounded or self.mj.viewer is not None:
            return
        self.mj.camera.distance -= value / 20
        
    def release_mouse_cb(self):
        self.last = None

    def drag_cb(self, sender, value):
        if self.modal_visible(): return
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
        self.mj.perturb_camera(dx, dy)
    
    def pause(self):
        """
        Pauses the simulation
        """
        if self.mj.running_event.is_set():
            self.mj.running_event.clear()
        else:
            self.mj.running_event.set()

    def reset(self):
        """
        Resets the simulation
        """
        self.mj.reset_event.set()

    def hard_reset(self):
        """
        Triggers a hard reset of the simulation. This will reload the map other race
        attributes such as the driver paths and names.
        """
        self.mj.hard_reset_event.set()
    
    def focus_on_next_car(self):
        """
        Locks the camera to the next car in the sequence. Cycles at the end
        """
        if len(self.mj.vehicle_states) > 0:
            if self.mj.option("sort_vehicle_list"):
                positions = { m.id : i for i, m in enumerate(reversed(sorted(self.mj.vehicle_states, key=lambda v: v.absolute_completion()))) }
                rpositions = { i : m.id for i, m in enumerate(reversed(sorted(self.mj.vehicle_states, key=lambda v: v.absolute_completion()))) }
                self.mj.watching = rpositions[(positions[self.mj.watching or 0] - 1) % len(self.mj.vehicle_states)]
            else:
                self.mj.watching = ((self.mj.watching or 0) + 1) % len(self.mj.vehicle_states)

    def focus_on_previous_car(self):
        """
        Locks the camera to the previous car in the sequence. Cycles at the end
        """
        if len(self.mj.vehicle_states) > 0:
            if self.mj.option("sort_vehicle_list"):
                positions = { m.id : i for i, m in enumerate(reversed(sorted(self.mj.vehicle_states, key=lambda v: v.absolute_completion()))) }
                rpositions = { i : m.id for i, m in enumerate(reversed(sorted(self.mj.vehicle_states, key=lambda v: v.absolute_completion()))) }
                self.mj.watching = rpositions[(positions[self.mj.watching or 0] + 1) % len(self.mj.vehicle_states)]
            else:
                self.mj.watching = ((self.mj.watching or 0) - 1) % len(self.mj.vehicle_states) 

    def toggle_cinematic_camera(self):
        """
        Toggle the cinemaic camera value. If cinematic mode is turned off as a result,
        the camera's velocity is set to zero.
        """
        self.mj.option("cinematic_camera", not self.mj.option("cinematic_camera"))
        if not self.mj.option("cinematic_camera"):
            self.mj.camera_vel[0] = 0
            self.mj.camera_vel[1] = 0

    def press_key_cb(self, sender, keycode):
        if self.modal_visible(): return
        command = self.keybindings.get(keycode) or self.keybindings.get(chr(keycode))
        if command is None:
            if chr(keycode) == "A":
                self.steering_angle = 1.0
            elif chr(keycode) == "D":
                self.steering_angle = -1.0
            elif chr(keycode) == "W":
                self.speed = self.mj.option("manual_control_speed")
            elif chr(keycode) == "S":
                self.speed = -self.mj.option("manual_control_speed")
            else:
                # print(f"Unbound keycode {keycode}")
                pass

    def modal_visible(self):
        for modal in ["cars modal", "keybindings modal"]:
            try:
                if dpg.get_item_configuration(modal)["show"]:
                    return True
            except:
                pass
        return False

    def release_key_cb(self, sender, keycode):
        if self.modal_visible(): return
        command = self.keybindings.get(keycode) or self.keybindings.get(chr(keycode))
        if command is not None:
            command = self.commands[command]
            print(command.tag)
            command.callback()
        else:
            if chr(keycode) in ["A", "D"]:
                self.steering_angle = 0.0
            elif chr(keycode) in ["W", "S"]:
                self.speed = 0.0
            else:
                pass
        
    def run(self):
        global exit_event
        print("GUI thread spawned")
        dpg.create_context()
        with dpg.item_handler_registry(tag="map_combo_handler_registry"):
            dpg.add_item_clicked_handler(0, callback=self.tracks_combo_clicked_cb)
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.scroll_cb)
            dpg.add_mouse_release_handler(callback=self.release_mouse_cb)
            dpg.add_mouse_drag_handler(callback=self.drag_cb)
            dpg.add_key_release_handler(callback=self.release_key_cb)
            dpg.add_key_press_handler(callback=self.press_key_cb)
        with dpg.texture_registry(show=False):
            format = dpg.mvFormat_Float_rgba if platform == "darwin" else dpg.mvFormat_Float_rgb
            dpg.add_raw_texture(1920, 1080, self.pixels, format=format, tag="visualisation")
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
                    with dpg.table(header_row=False):
                        dpg.add_table_column()
                        dpg.add_table_column()
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_button(label="Reset", callback=self.reset, width=-1)
                            dpg.add_button(label="Hard Reset", callback=self.hard_reset, width=-1)
                            dpg.add_button(label="Pause", callback=self.pause, width=-1)
                    dpg.add_checkbox(label="External Viewer (faster)",
                                     callback=lambda sender, value: self.set_inline_panel_visibility(not value))
                    dpg.add_combo(tag="Tracks Combo",
                                  default_value=self.mj.map_metadata["name"],
                                  label="map",
                                  callback=self.select_map_cb)
                    dpg.add_separator()
                    dpg.add_button(label="Vehicle Info", callback=self.show_cars_modal, width=-1)
                    with dpg.collapsing_header(label="Vehicle Overview", tag="dashboard", default_open=True):
                        pass
                    dpg.add_separator()
                    with dpg.collapsing_header(label="Options", tag="Options", default_open=True):
                        dpg.add_button(label="Commands Info", callback=self.show_keybindings_modal, width=-1)
                        dpg.add_separator()
                        pass
        dpg.bind_item_handler_registry("Tracks Combo", "map_combo_handler_registry")
        dpg.create_viewport(title="Formula Trinity AI Grand Prix", width=self.window_size[0], height=self.window_size[1])
        dpg.set_viewport_resize_callback(self.viewport_resize_cb)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        
        last = time.time()
        while dpg.is_dearpygui_running():
            if exit_event.is_set():
                break
            now = time.time()
            if (now - last < 1/self.mj.option("max_fps")):
                time.sleep(1/self.mj.option("max_fps") - (now - last));
            # print (f"{1/(time.time()-last)} fps")
            last = time.time()
            self.inject_vehicle_state([vehicle_state.id for vehicle_state in self.mj.vehicle_states])
            self.inject_options(self.mj.options)
            if self.viewport_resize_event.is_set():
                self.viewport_resize_cb(None, [*self.window_size, *self.window_size])
                self.viewport_resize_event.clear()
            dpg.render_dearpygui_frame()
        exit_event.set()
        dpg.destroy_context()
    
    def succ_keys(self, sender, keycode, command):
        readable = readable_keycode(keycode)
        conflict = self.keybindings.get(keycode) or self.keybindings.get(readable)
        if conflict is None or conflict == command.tag:
            dpg.configure_item("keybindings select", show=False)
            dpg.configure_item("keybindings dashboard", show=True)
            dpg.delete_item("keybindings handler")
            existing_keycode = invert(self.keybindings).get(command.tag)
            if existing_keycode is not None:
                del self.keybindings[existing_keycode]
            if keycode == 256:
                dpg.configure_item(f"keybinding for {command.tag}", label=f"N/A")
            else:
                # TODO: store in unreadable form  …
                self.keybindings[keycode] = command.tag
                dpg.configure_item(f"keybinding for {command.tag}", label=f"`{readable}`")
        else:
            command = self.commands[conflict]
            dpg.set_value("keybindings message", f"`{readable}` would clash with command `{command.tag}`")

    def supplant(self, sender, value, command):
        dpg.configure_item("keybindings dashboard", show=False)
        dpg.configure_item("keybindings select", show=True)
        dpg.set_value("keybindings target", f"Setting keybinding for `{command.tag}`")
        with dpg.handler_registry(tag="keybindings handler"):
            dpg.add_key_release_handler(callback=self.succ_keys, user_data=command)
    
    def show_cars_modal(self):
        """
        Shows the user a dialog they can use to configure the race. The can configure the
        number of cars involved and the attributes of each car.
        """
        @dataclass
        class Tags:
            path_id : int = tag()
            name_id  : int = tag()
            icon_id : int = tag()
            primary_id : int = tag()
            secondary_id : int = tag()
            texture_id : int = tag()
            image_id : int = tag()
            icon_group_id : int = tag()
            
        def aggregate_cars_from_ui():
            cars = []
            for child in dpg.get_item_children("cars", 1):
                tags = dpg.get_item_configuration(child)["user_data"]
                car = {
                    "driver" : dpg.get_value(tags.path_id),
                    "name" : dpg.get_value(tags.name_id),
                    "icon" : dpg.get_value(tags.icon_id),
                    # FIXME: support transparency in car colors
                    "primary" : dpg.get_value(tags.primary_id)[:3],
                    "secondary" : dpg.get_value(tags.secondary_id)[:3]
                }
                cars.append(car)
            return cars

        def apply_cb(sender, value, user_data):
            self.mj.nuke("cars_path")
            self.mj.cars = aggregate_cars_from_ui()
            self.hard_reset()

        def write_cb(sender, value, user_data):
            self.mj.nuke("cars_path")
            basename = dpg.get_value("cars output")
            if basename == "":
                print("Cant write to empty basename")
                return
            cars = aggregate_cars_from_ui()
            path = os.path.join("template", "cars", basename)
            with open(path, "w") as file:
                json.dump(cars, file)

        def clear():
            dpg.delete_item("cars", children_only=True)

        def load(cars):
            clear()
            for car in cars:
                tags = new_car_cb(None, None)
                dpg.set_value(tags.name_id, car["name"])
                dpg.set_value(tags.path_id, car["driver"])
                dpg.set_value(tags.icon_id, car["icon"])
                icon_cb(None, car["icon"], [tags.texture_id, tags.icon_id])
                dpg.set_value(tags.primary_id, resolve_color(car["primary"]))
                dpg.set_value(tags.secondary_id, resolve_color(car["secondary"]))
        
        def load_from_existing_cb(sender, value):
            # FIXME: Exception is raised in python and not handled
            # right now if the JSON is invalid
            try:
                with open(os.path.join("template", "cars", value)) as cars_file:
                    cars = json.load(cars_file)
            except Exception as e:
                print(e)
                return
            load(cars)
            dpg.set_value("cars output", value)
                
        def icon_cb(sender, icon_basename, user_data):
            texture_tag, image_tag = user_data
            try:
                image_path = os.path.join("template", "icons", icon_basename)
                image = Image.open(image_path).convert("RGB").resize((100, 100))
            except Exception as e:
                print(e)
                return
            dpg.set_value(texture_tag, np.array(image, dtype=np.float32) / 255.0)
            dpg.configure_item(image_tag, show=True)
            
        def delete_car_cb(sender, value, group):
            dpg.delete_item(group)

        def move_car_up_cb(sender, value, group):
            children = dpg.get_item_children("cars", 1)
            index = children.index(group)
            try:
                children[index-1], children[index] = children[index], children[index-1]
                dpg.reorder_items("cars", 1, children)
            except IndexError:
                pass

        def move_car_down_cb(sender, value, group):
            children = dpg.get_item_children("cars", 1)
            index = children.index(group)
            try:
                children[index], children[index+1] = children[index+1], children[index]
                dpg.reorder_items("cars", 1, children)
            except IndexError:
                pass

        def import_car_cb(sender, value):
            dpg.configure_item("import cars button", show=False)
            dpg.configure_item("import cars combo", show=True)

        def finish_import_car_cb(sender, value):
            dpg.configure_item("import cars button", show=True)
            dpg.configure_item("import cars combo", show=False)
            value = ".".join(value.split(".")[1:])
            with open(os.path.join("drivers", f"{value}.json")) as f:
                driver = json.load(f)
            cars = [*aggregate_cars_from_ui(), driver]
            load(cars)
            
        def new_car_cb(sender, value):
            tags = Tags()
            with dpg.group(parent="cars", user_data=tags) as group:
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_input_text(tag=tags.name_id, label="Driver Name")
                        dpg.add_combo(tag=tags.path_id, label="Driver Path")
                        dpg.add_combo(tag=tags.icon_id, label="Icon", callback=icon_cb,
                                           user_data=[tags.texture_id, tags.image_id])
                        dpg.add_color_edit(tag=tags.primary_id, label="Primary")
                        dpg.add_color_edit(tag=tags.secondary_id, label="Secondary")
                    with dpg.group():
                        with dpg.group():
                            dpg.add_button(label="Up", callback=move_car_up_cb, user_data=group, width=40)
                            dpg.add_button(label="Dn", callback=move_car_down_cb, user_data=group, width=40)
                        with dpg.group(tag=tags.icon_group_id):
                            pass
                dpg.add_button(label="Delete", callback=delete_car_cb, user_data=group, width=-1)
                dpg.add_separator()
                # memory leak here
                with dpg.item_handler_registry():
                    dpg.add_item_clicked_handler(0, callback=self.driver_path_combo_clicked_cb, user_data=tags)
                dpg.bind_item_handler_registry(tags.path_id, dpg.last_container())
                with dpg.item_handler_registry():
                    dpg.add_item_clicked_handler(0, callback=self.icon_clicked_cb, user_data=tags)
                dpg.bind_item_handler_registry(tags.icon_id, dpg.last_container())
            dpg.add_raw_texture(100, 100, np.zeros((100, 100, 3), dtype=np.float32),
                                format=dpg.mvFormat_Float_rgb,
                                tag=tags.texture_id, parent="car icons")
            dpg.add_image(tags.texture_id, width=40, height=40, show=True,
                          tag=tags.image_id,
                          parent=tags.icon_group_id)
            return tags
        if dpg.does_item_exist("cars modal"):
            dpg.delete_item("cars modal")
        if dpg.does_item_exist("car icons"):
            dpg.delete_item("car icons")
        if dpg.does_item_exist("import_cars_combo_handler_registry"):
            dpg.delete_item("import_cars_combo_handler_registry")
        dpg.add_texture_registry(tag="car icons")
        with dpg.window(tag="cars modal", width=400, height=400, no_scrollbar=True):
            with dpg.table(header_row=False):
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_button(label="New Car", callback=new_car_cb, width=-1)
                    with dpg.group():
                        dpg.add_button(tag="import cars button", label="Import Car", callback=import_car_cb, width=-1)
                        dpg.add_combo(tag="import cars combo", width=-1, callback=finish_import_car_cb, show=False)
                        with dpg.item_handler_registry(tag="import_cars_combo_handler_registry"):
                            dpg.add_item_clicked_handler(0, callback=self.import_cars_combo_clicked_cb)
                        dpg.bind_item_handler_registry("import cars combo", "import_cars_combo_handler_registry")
                    dpg.add_button(label="Load from Sim", callback=lambda: load(self.mj.cars), width=-1)
            dpg.add_button(label="Clear", callback=clear, width=-1)
            dpg.add_combo(
                label="Load from file",
                items=os.listdir(os.path.join("template", "cars")),
                callback=load_from_existing_cb
            )
            dpg.add_separator()
            with dpg.child_window(tag="cars", height=-75):
                pass
            with dpg.group():
                dpg.add_input_text(tag="cars output", label="Output Name")
                dpg.add_button(label="Write", callback=write_cb, width=-1)
                dpg.add_button(label="Apply", callback=apply_cb, width=-1)
                dpg.add_text(tag="cars error", color=colors["error"], show=False)

    def show_keybindings_modal(self):
        """
        Presents a window with all the commands and their associated keybindings, if any
        """
        inverted_keybindings_index = invert(self.keybindings)
        # FIXME: Exiting the keybindings screen early disables all
        # further key presses as the callback isn't restored
        # correctly.
        if dpg.does_item_exist("keybindings modal"):
            dpg.delete_item("keybindings modal")
        with dpg.window(tag="keybindings modal"):
            with dpg.group(tag="keybindings select", show=False):
                dpg.add_text(tag="keybindings target")
                dpg.add_text("waiting for user input ...", tag="keybindings message", color=colors["silver"])
                dpg.add_text("(press escape to unbind)")
            with dpg.group(tag="keybindings dashboard"):
                for command in self.commands.values():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=command.label + ":",
                                       callback=lambda sender, value, command: command.callback(),
                                       user_data=command)
                        key = inverted_keybindings_index.get(command.tag)
                        if key is None:
                            label = "N/A"
                        else:
                            label = f"`{key}`"
                        dpg.add_button(tag=f"keybinding for {command.tag}",
                                       label=label,
                                       callback=self.supplant,
                                       user_data=command)
                    if command.description is not None and command.description != "":
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=20)
                            with dpg.group():
                                description = re.sub(r"\s+", " ", command.description.strip())
                                dpg.add_text(description, wrap=350, color=colors["silver"])

    def viewport_resize_cb(self, value, app):
        if self.window_size == app[2:]:
            return
        self.window_size = app[2:]
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
        if self.mj.viewer is None:
            self.mj.start_inline_render_thread()

    def icon_clicked_cb(self, sender, value, tags):
        dpg.configure_item(tags.icon_id, items=os.listdir(os.path.join(self.mj.template_dir, "icons")))

    def driver_path_combo_clicked_cb(self, sender, value, tags):
        items = [
            "ft_grandprix.fast",
            "ft_grandprix.nidc",
            "ft_grandprix.lobotomy"
        ]
        for file in os.listdir("drivers"):
            if ".py" not in file or "__" in file:
                continue
            items.append(f"drivers.{file[:-3]}")
        dpg.configure_item(tags.path_id, items=items)
        
    def import_cars_combo_clicked_cb(self):
        items = []
        # the drivers path should just be the location where you
        # extract stuff from google drive
        drivers_path = "drivers"
        try:
            compute_driver_files(drivers_path, silent=True)
        except Exception as e:
            print("error computing driver files:", e)
        for file in os.listdir(drivers_path):
            if ".json" not in file or "__" in file:
                continue
            items.append(f"drivers.{file[:-5]}")
        dpg.configure_item("import cars combo", items=items)

    def tracks_combo_clicked_cb(self):
        items = [".".join(os.path.basename(f).split(".")[:-1])
                 for f in os.listdir(self.mj.template_dir)
                 if "svg" not in f and "png" in f]
        dpg.configure_item("Tracks Combo", items=items)
        
    def select_map_cb(self, sender, track):
        self.mj.track = track
        self.mj.hard_reset_event.set()

# A simple JSON option that can be persisted in the file system
class Option:
    def __init__(self, tag, default, _type=None, callback=None, description=None, data=None, label=None, persist=True, min_value=None, max_value=None, present=True):
        self.tag = tag
        self.dpg_tag = f"__option__::{tag}"
        self.dpg_description_tag = f"__option__::{tag}::__description__"
        self.label = label or tag
        self.persist = persist
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.value = default
        self.present = present
        self.callback = callback
        self.type = _type or type(default)
        if self.persist and self.tag in data:
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
        self.hard_reset_event = Event()
        self.running_event = Event()
        self.launch_viewer_event = Event()
        self.kill_viewer_event = Event()
        self.mv = mv
        self.camera_vel = [0, 0]
        self.camera_pos_vel = [0, 0, 0]
        self.camera_friction = [0.01, 0.01]
        self.watching = 0
        self.rendered_dir = "rendered"
        self.template_dir = "template"
        self._camera = mujoco.MjvCamera()
        self.options = {}
        self.cars = []
        self.track = track
        
        try:
            with open("aigp_settings.json") as data_file:
                data = json.load(data_file)
        except Exception as e:
            data = {}
            print("Not loading settings from file: ", e)
        
        self.declare("sort_vehicle_list", False, label="Sort Vehicles", description="Keeps the vehicle list sorted by position", data=data)
        self.declare("reset_camera", True, label="Reset Camera", data=data,
                     description="Resetting sets sets the camera position ot the first path point")
        self.declare("option_intensity", 1.0, label="Icon Intensity", data=data, callback=self.set_icon_intensity)
        self.declare("lock_camera", False, label="Lock Camera", data=data,
                     description="If this option is set, the camera angle will be kept in line with the angle of the vehicle being watched")
        self.declare("detach_control", False, label="Detached Control", data=data,
                     description="Do not attempt to deliver controls to the car being watched")
        self.declare("manual_control", False, label="Manual Control", persist=False,
                     description="Control the car being watched with the W, A, S and D keys")
        self.declare("always_invoke_driver", True, data=data, label="Invoke with Manual", present=False,
                     description="Will keep on invoking the process LiDAR function even if manual control is set")
        self.declare("manual_control_speed", 3.0, label="Manual Control Speed", data=data,
                     description="The speed at which the car will drive when being controlled manually")
        self.declare("cars_path", "cars.json", _type=str, label="Cars Path", persist=False)
        self.declare("lap_target", 10, data=data, label="Lap Target")
        self.declare("max_fps", 30, data=data, label="Max FPS")
        self.declare("cinematic_camera", False, data=data, label="Cinematic Camera")
        self.declare("center_camera", False, data=data, label="Center Camera")
        self.declare("center_camera_inside", True, data=data, label="Center Camera Inside", present=False,
                     description="If this option is set to true, the 'center_camera' option will view the vehicle being watched from the inside of the track")
        self.declare("pause_on_reload", True, data=data, label="Pause on Reload")
        self.declare("save_on_exit", True, data=data, label="Save on Exit", present=False,
                     description="Save to `aigp_settings.json` on exit")
        self.declare("bubble_wrap", False, label="Soften Collisions", data=data, persist=True,
                     description="Soften collisions with the map border so that vehicles don't get stuck as easilly",
                     callback=self.soften)
        self.declare("physics_fps", 500, data=data, label="Max Physics FPS", present=False,
                     description="The reciprocal of the physics timestep passed into mujoco")
        self.declare("max_geom", 1500, data=data, label="Mujoco Geom Limit", persist=False, present=False,
                     description="The number of entities that mujoco will render")
        self.declare("rangefinder_alpha", 0.1, label="Rangefinder Intensity", data=data, callback=self.rangefinder)
        self.declare("tricycle_mode", False, label="Use a Tricycle 🤡", data=data, present=False, persist=True,
                     description="Use the old differential drive vehicle model (requires hard reset)",
                     callback=lambda x: self.hard_reset_event.set())
        self.declare("naive_flatten", False, label="Naive Flatten", data=data, present=False, persist=True,
                     description="Prevent vehicles from rotating in a naive manner. Violates some constraints of the physics engine and may lead to cars flying away.")
        self.declare("debug_mode", False, label="Debug Mode", data=data,
                     description="Shows hidden debugging settings")
        self.declare("map_color", [1, 0, 0, 1], label="Map Color", data=data, present=False)
        self.declare("rangefinder_tilt", 0.0, label="Rangefinder Tilt", data=data, present=False, min_value=0, max_value=np.pi)
        self.declare("use_simulated_simulation_lidar", False, label="Simulate Lidar", data=data, present=False,
                     description="If this is used, mujoco's slow raycasting will be avoided and raycasting will instead be performed using the 2D image and vehicle size and position metadata (fater)",
                                  callback=self.set_use_simulated_simulation_lidar_flag)

        self.vehicle_states = []
        self.shadows = {}
        self.steps = 0

        self.viewer = None
        self.kill_inline_render_event = Event()
        self.render_finished = Event()
        self.render_finished.set()

    def set_use_simulated_simulation_lidar_flag(self, flag):
        if flag:
            for vehicle_state in self.vehicle_states:
                self.shadow_rangefinders(vehicle_state.id)
        else:
            for vehicle_state in self.vehicle_states:
                if vehicle_state.id not in self.shadows:
                    self.unshadow_rangefinders(vehicle_state.id)
            

    def set_icon_intensity(self, intensity):
        for i in range(len(self.vehicle_states)):
            self.model.mat(f"car #{i} icon").rgba[:4] = intensity

    @property
    def camera(self):
        if self.viewer is None or type(self.viewer) is str:
            return self._camera
        else:
            return self.viewer.cam

    def perturb_camera_pos(self, dx, dy, dz):
        if self.option("cinematic_camera"):
            if self.watching is not None:
                target = self.data.body(f"car #{self.watching}").xpos[:]
                delta = target - self.camera.lookat
                delta_magnitude = np.linalg.norm(delta)
                next_lookat = target - (self.camera.lookat + self.camera_pos_vel)
                next_lookat_magnitude = np.linalg.norm(next_lookat)
                P = 0.1 * delta
                D = 0.05 * (next_lookat_magnitude - delta_magnitude) * delta
                self.camera_pos_vel = P + D

    def perturb_camera(self, dx, dy):
        if not self.option("cinematic_camera"):
            self.camera.azimuth   += dx
            self.camera.elevation += dy
        else:
            self.camera_vel[0] += dx / 100
            self.camera_vel[1] += dy / 100

    def soften(self, soften=True):
        try:
            if self.mushr:
                for vehicle_state in self.vehicle_states:
                    self.model.geom(f"bl softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"br softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"fr softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"fl softener #{vehicle_state.id}").conaffinity = int(soften) << 2
            else:
                for vehicle_state in self.vehicle_states:
                    self.model.geom(f"left softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"right softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"front softener #{vehicle_state.id}").conaffinity = int(soften) << 2
        except Exception as e:
            print("Error configuring collision softening geoms.", e)

    def run(self):
        self.stage()
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
            os.remove(path)

    def declare(self, tag, default, **kwargs):
        self.options[tag] = Option(tag, default, **kwargs)

    def nuke(self, tag):
        self.options[tag].value = None

    def option(self, tag, value=None):
        option = self.options[tag]
        if value is not None:
            # print(f"`{tag}` = `{value}`")
            option.value = value
            if option.callback is not None:
                option.callback(value)
        return option.value
    
    def reload(self):
        if self.option("pause_on_reload"):
            self.running_event.clear()
        mujoco.mj_resetData(self.model, self.data)
        for m in self.vehicle_states:
            self.unshadow(m.id)
        vehicle_states = []
        for i, car in enumerate(self.mjcf_metadata["cars"]):
            if car["driver"].startswith("file://"):
                path = car["driver"][7:-3].replace("/", ".")
            elif "//" not in car["driver"]:
                # if no protocol is given, assume module import is
                # being used
                path = car["driver"]
            else:
                print("Unsupported schema: supported (file://)")
            print(f"Loading driver from python module path '{path}'")
            try:
                driver = runtime_import(path).Driver()
            except:
                driver = LobotomyDriver()
            vehicle_state = VehicleState(
                id           = i,
                offset       = (i+5) * 2,
                driver_path  = path,
                driver       = driver,
                label        = car["name"],
                data         = self.data,
                rangefinders = self.mjcf_metadata["rangefinders"]
            )
            vehicle_states.append(vehicle_state)
        self.vehicle_states = vehicle_states
        if self.watching is not None and self.watching >= len(self.vehicle_states):
            self.watching = None
        self.shadows = {}
        self.steps = 0
        self.winners = {}
        if self.option("reset_camera"):
            self.camera.lookat[:2] = self.path[0]
        self.position_vehicles(self.path)

    def rangefinder(self, value):
        self.model.vis.rgba.rangefinder[3] = value
    
    def stage(self, track=None):
        if track is None:
            if self.track is None:
                raise RuntimeError("stage must be called with a track first")
        else:
            self.track = track
        if self.option("cars_path") is not None:
            cars_path = os.path.join(self.template_dir, "cars", self.option("cars_path"))
            try:
                with open(cars_path) as cars_file:
                    self.cars = json.load(cars_file)
            except Exception:
                print(f"ERROR: Could not read from `{cars_path}`")
                self.cars = []
        
        image_path = os.path.join(self.template_dir, f"{self.track}.png")
        balls = np.array(Image.open(image_path))
        balls[balls != 255] = 0
        balls = 255 - balls
        # from scipy.ndimage import distance_transform_edt
        # self.dt = distance_transform_edt(balls)
        if not self.option("tricycle_mode"):
            chunk(image_path, verbose=False, force=True, scale=2.0)
            produce_mjcf(
                template_path=os.path.join(self.template_dir, "mushr.em.xml"),
                rangefinders=90,
                cars=self.cars,
                map_color=self.option("map_color")[:3]
            )
            self.mushr = True
        else:
            chunk(image_path, verbose=False, force=True, scale=2.0)
            produce_mjcf(
                template_path=os.path.join(self.template_dir, "car.em.xml"),
                rangefinders=90,
                cars=self.cars
            )
            self.mushr = False
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
        self.path[:, 0] =   self.path[:, 0] / self.map_metadata['width']  * self.map_metadata['chunk_width'] * self.map_metadata['scale']
        self.path[:, 1] = - self.path[:, 1] / self.map_metadata['height'] * self.map_metadata['chunk_height']  * self.map_metadata['scale']
        self.restart_render_thread()
        self.reload()
        if platform == "darwin":
            for vehicle_state in self.vehicle_states:
                id = vehicle_state.id
                self.model.light(f"top light #{id}").active = 0
                self.model.light(f"front light #{id}").active = 0
        self.sync_options()

    def sync_options(self):
        """
        Invokes any callbacks associated with options and ensure that the application state
        is valid this can be considered generally safe as options are ready to be set to
        any value at any time in a thread-safe manner.
        """
        for option in self.options.values():
            if option.callback is not None:
                option.callback(option.value)

    def physics(self):
        Thread(target=self.physics_thread).start()

    def restart_render_thread(self):
        """
        Restarts the external viewer thread, or the internal rendering thread, depending
        on which one is currently visible.
        """
        if self.viewer is None:
            if not self.render_finished.is_set():
               self.kill_inline_render_event.set()
               self.render_finished.wait()
            Thread(target=self.inline_render_thread).start()
        else:
            self.launch_viewer_event.set()

    def start_inline_render_thread(self):
        if self.viewer is None:
            if not self.render_finished.is_set():
                self.kill_inline_render_event.set()
                self.render_finished.wait()
            Thread(target=self.inline_render_thread).start()

    def reload_code(self, index):
        self.vehicle_states[index].reload_code()

    def position_vehicles(self, path=None):
        """
        Sends all vehicles back to their starting positions without changing race
        metadata such as number of laps, lap times .etc.
        """
        if path is None:
            path = self.path
        for i, m in enumerate(self.vehicle_states):
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
            if self.option("lock_camera") and self.watching is not None:
                quat = self.vehicle_states[self.watching].joint.qpos[3:]
                desired_azimuth = quaternion_to_angle(*quat)
                camera_azimuth = self.camera.azimuth % 360
                desired_azimuth = math.degrees(desired_azimuth) % 360
                delta = (desired_azimuth - camera_azimuth + 180) % 360 - 180
                self.perturb_camera(delta * 0.01, 0)
            if self.option("cinematic_camera"):
                self.camera.azimuth += self.camera_vel[0]
                self.camera.elevation += self.camera_vel[1]
                self.camera_vel[0] *= (1 - self.camera_friction[0])
                self.camera_vel[1] *= (1 - self.camera_friction[1])
                self.camera.lookat += self.camera_pos_vel
                self.camera_pos_vel[0] *= (1 - self.camera_friction[0])
                self.camera_pos_vel[1] *= (1 - self.camera_friction[0])
                self.camera_pos_vel[2] *= (1 - self.camera_friction[0])
                
            if self.launch_viewer_event.is_set():
                if self.viewer is not None:
                    viewer = self.viewer
                    viewer._get_sim().load(self.model, self.data, '')
                    # self.viewer.close()
                else:
                    self.viewer = "John the Baptist"
                    self.viewer = mujoco.viewer.launch_passive(
                        self.model,
                        self.data,
                        key_callback = lambda keycode: self.mv.release_key_cb(None, keycode),
                        show_left_ui = False,
                        show_right_ui = False
                    )
                    self.kill_inline_render_event.set()
                    self.mv.viewport_resize_event.set()
                self.launch_viewer_event.clear()
            if self.kill_viewer_event.is_set():
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                self.kill_viewer_event.clear()
                self.start_inline_render_thread()
                self.mv.viewport_resize_event.set()
            if self.watching is not None:
                target = self.data.body(f"car #{self.watching}").xpos[:]
                if self.option("cinematic_camera"):
                    self.perturb_camera_pos(*(target - self.camera.lookat))
                else:
                    self.camera.lookat[:] = target
                if self.option("center_camera"):
                    center = np.array([0.5 * self.map_metadata['chunk_width'] * self.map_metadata['scale'],
                                       -0.5 * self.map_metadata['chunk_height'] * self.map_metadata['scale']])
                    angle = np.arctan2(center[1] - target[1],
                                       center[0] - target[0])
                    # print(target, center, angle)
                    if self.option("center_camera_inside"):
                        self.camera.azimuth = math.degrees(angle) - 180
                    else:
                        self.camera.azimuth = math.degrees(angle)
            if self.reset_event.is_set():
                self.reload()
                self.reset_event.clear()
            if self.hard_reset_event.is_set():
                self.stage()
                self.hard_reset_event.clear()
            if exit_event.is_set():
                self.persist()
                print("Persisted!")
                if self.viewer is not None:
                    self.viewer.close()
                break
            
            if self.viewer is not None:
                self.viewer.cam.lookat = self.camera.lookat
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
            for vehicle_state in self.vehicle_states:
                if self.option("naive_flatten"):
                    vehicle_state.joint.qpos[3:] = euler_to_quaternion([quaternion_to_angle(*vehicle_state.joint.qpos[3:]), 0, 0])
                xpos = vehicle_state.joint.qpos[0:2]
                distances = ((self.path - xpos)**2).sum(1)
                closest = distances.argmin()
                vehicle_state.distance_from_track = distances[closest]
                vehicle_state.off_track = vehicle_state.distance_from_track > 1
                if not vehicle_state.off_track:
                    completion = (closest - vehicle_state.offset) % 100
                    delta = completion - vehicle_state.completion
                    vehicle_state.delta = (completion - vehicle_state.completion + 50) % 100 - 50
                    
                    if abs(delta) > 90:
                        lap_time = (self.steps - vehicle_state.start) * self.model.opt.timestep
                        if vehicle_state.delta < 0:
                            vehicle_state.laps -= 1
                            vehicle_state.good_start = False
                            if len(vehicle_state.times) != 0:
                                vehicle_state.times.pop()
                        elif vehicle_state.delta > 0:
                            if vehicle_state.good_start:
                                # dont add a new lap OR start counting
                                # time for the next lap if we went
                                # backwards, we are still in the current
                                # lap
                                vehicle_state.times.append(lap_time)
                                vehicle_state.start = self.steps
                            vehicle_state.laps += 1
                            vehicle_state.good_start = True
                    if vehicle_state.laps >= self.option("lap_target"):
                        if vehicle_state.id not in self.winners:
                            self.winners[vehicle_state.id] = len (self.winners) + 1
                        vehicle_state.finished = True
                        self.shadow(vehicle_state.id)
                    vehicle_state.completion = completion

                id = self.model.sensor("car #0 accelerometer").id
                d = self.data.sensordata[id:id+3]
                # print(f"ACCEL: data: {d} accel: {math.sqrt((d**2).sum())}")
                id = self.model.sensor("car #0 gyro").id
                d = self.data.sensordata[id:id+3]
                # print(f"GYRO:  accel: {math.sqrt((d**2).sum())}")

                if self.option("use_simulated_simulation_lidar"):
                    s = (20 * self.map_metadata["scale"])
                    i_x = (xpos[0] / s) * self.map_metadata["original_width"]
                    i_y = -(xpos[1] / s) * self.map_metadata["original_height"]
                    print(i_x, i_y)
                    r = 90
                    angles = np.linspace(self.option("rangefinder_tilt") + snapshot.yaw+np.pi, snapshot.yaw-np.pi, r, endpoint=False)
                    s = time.time()
                    ranges, points = fakelidar(i_x, i_y, self.dt, r, np.cos(angles), np.sin(angles))
                    e = time.time()
                    print("TIME: ", e - s)
                    ranges /= self.map_metadata["original_width"]
                    ranges *= s
                else:
                    ranges = self.data.sensordata[vehicle_state.sensors]

                snapshot = vehicle_state.snapshot(time = self.steps / self.model.opt.timestep)
                if vehicle_state.v2: driver_args = [ranges, snapshot]
                else:                driver_args = [ranges]

                speed, steering_angle = 0.0, 0.0

                if self.option("always_invoke_driver") or not self.option("manual_control"):
                    # TODO: run this in its own thread so that the
                    # user can do things like time.sleep() without
                    # blocking the executor
                    try:
                        speed, steering_angle = vehicle_state.driver.process_lidar(*driver_args)
                    except Exception as e:
                        print(f"Error in vehicle `{vehicle_state.label}`: `{e}`")
                        continue
                    
                if self.option("manual_control") and self.watching == vehicle_state.id:
                    speed, steering_angle = self.mv.speed, self.mv.steering_angle
                    if speed == 0.0 and self.data.ctrl[vehicle_state.forward] > 0.0:
                        speed = self.data.ctrl[vehicle_state.forward] * 0.99
                
                vehicle_state.speed = speed
                vehicle_state.steering_angle = steering_angle
                
                if not self.option("detach_control"):
                    self.data.ctrl[vehicle_state.forward] = vehicle_state.speed
                    self.data.ctrl[vehicle_state.turn] = vehicle_state.steering_angle
                
            mujoco.mj_step(self.model, self.data)
            self.steps += 1

            p_fps = self.option("physics_fps")
            now = time.time()
            fps = 1 / (now - last) if now - last > 0 else p_fps
            if (now - last < 1/p_fps):
                time.sleep(1/p_fps - (now - last));
            last = time.time()
            # print(f"{fps} fps")

    def shadow_rangefinders(self, i):
        for j in range(self.mjcf_metadata["rangefinders"]):
            self.model.sensor(f"rangefinder #{i}.#{j}").type = mujoco.mjtSensor.mjSENS_USER

    # send car to the shadow realm
    def shadow(self, i):
        if i in self.shadows:
            return self.shadows[i]
        vehicle_state = self.vehicle_states[i]
        # lobotomise car first
        vehicle_state.driver = LobotomyDriver()
        vehicle_state.v2 = False
        vehicle_state.driver_path = "ft_grandprix.lobotomy"
        shadows = []
        self.shadow_rangefinders(i)
        transparent_material_id  = self.model.mat("transparent").id
        shadow_alpha = 0.1
        for mat in [f"car #{i} primary", f"car #{i} secondary", f"car #{i} body", f"car #{i} wheel", f"car #{i} icon"]:
            self.model.mat(mat).rgba[3] = shadow_alpha
        for geom in self.subgeoms(i):
            m, d = self.model.geom(geom), self.data.geom(geom)
            r = [*self.model.mat(m.matid.item()).rgba[:3], shadow_alpha]
            shadow = dict(type=m.type.item(), size=m.size, pos=d.xpos, rgba=r, mat=d.xmat)
            # turn off collisions (only makes sense wrt car.em.xml condim and
            # conaffinity values)
            m.conaffinity = 0
            m.contype = 1 << 1
            m.matid = transparent_material_id
            shadows.append(shadow)
        self.shadows[vehicle_state.id] = shadows
        return self.shadows[vehicle_state.id]

    def subgeoms(self, i):
        if self.mushr:
            return [f"chasis #{i}",
                    f"car #{i} lidar",
                    f"buddy_wheel_fl_throttle #{i}",
                    f"buddy_wheel_fr_throttle #{i}",
                    f"buddy_wheel_bl_throttle #{i}",
                    f"buddy_wheel_br_throttle #{i}"]
        else:
            return [f"chasis #{i}",
                    f"car #{i} lidar",
                    f"front wheel #{i}",
                    f"left wheel #{i}",
                    f"right wheel #{i}"]

    def unshadow_rangefinders(self, i):
        for j in range(90):
            self.model.sensor(f"rangefinder #{i}.#{j}").type = mujoco.mjtSensor.mjSENS_RANGEFINDER

    def unshadow(self, i):
        if i not in self.shadows:
            return
        self.unshadow_rangefinders(i)
        for mat in [f"car #{i} primary", f"car #{i} secondary", f"car #{i} body", f"car #{i} wheel", f"car #{i} icon"]:
            self.model.mat(mat).rgba[3] = 1.0
        for geom in self.subgeoms(i):
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
        if width * height == 0:
            print("Window area is zero, not going to render anything")
            self.render_finished.set()
            self.kill_inline_render_event.clear()
            return
        self.renderer = Renderer(self.model, width=width, height=height, max_geom=self.option("max_geom"))
        vopt = mujoco.MjvOption()
        scene = self.renderer.scene
        buf = np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
        while True:
            self.renderer.update_scene(self.data, self.camera, vopt)
            for shadows in self.shadows.values():
                for shadow in shadows:
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], **shadow)
                    self.renderer.scene.geoms[scene.ngeom].dataid = 0
                    scene.ngeom += 1
            self.renderer.render(out=buf)
            self.mv.pixels[0:height, 0:width, :3] = buf / 255.0
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

def display_top(snapshot, key_type='lineno', limit=100):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def profile():
    print("Profiling thread started")
    tracemalloc.start()
    while True:
        time.sleep(5)
        snapshot = tracemalloc.take_snapshot()
        print("\n---\n\n")
        display_top(snapshot)

# Thread(target=profile).start()

if __name__ == "__main__":
    try:
        ModelAndView(track="track").run()
    except KeyboardInterrupt:
        exit_event.set()
        pass

    
