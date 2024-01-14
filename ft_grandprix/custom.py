# TODO: only render on user input when the thing is paused

# TODO: consider not using integer meta-keys and identifying cars
# using some means that is actually unique.

import os
import re
from PIL import Image
import tempfile
import shutil
import math
import importlib
import queue
import json
from dataclasses import dataclass, field
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

def quaternion_to_angle(w, x, y, z):
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
        self.viewport_resize_event = Event()
        self.window_size = width, height
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
                Command("show_keybindings_modal", self.show_keybindings_modal)
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
        self.mj.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] ^= 1,

    def inject_options(self, options):
        for option in options.values():
            if not option.present:
                continue
            if dpg.does_item_exist(option.dpg_tag):
                dpg.set_value(option.dpg_tag, option.value if option.value is not None else option.type())
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
                            dpg.configure_item(option.tag, min_value=option.min_value)
                        if option.max_value:
                            dpg.configure_item(option.tag, max_value=option.max_value)
                    elif option.type is float:
                        dpg.add_input_float(**default_attributes)
                        if option.min_value:
                            dpg.configure_item(option.tag, min_value=option.min_value)
                        if option.max_value:
                            dpg.configure_item(option.ntag, max_value=option.max_value)
                    elif option.type is str:
                        if default_attributes["default_value"] is None:
                            default_attributes["default_value"] = ""
                        dpg.add_input_text(**default_attributes)
                    if option.description is not None:
                        description = re.sub(r"\s+", " ", option.description.strip())
                        with dpg.group(horizontal=True):
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
    
    def inject_meta(self, meta):
        @dataclass
        class Tags:
            completion_id    : int = tag()
            laps_id          : int = tag()
            position_id      : int = tag()
            times_id         : int = tag()
            pending_id       : int = tag()
            finished_id      : int = tag()
            finished_text_id : int = tag()
            tag              : int = tag()
        mj_meta = self.mj.meta # hack to make sure that this is atomic
        tags = [dpg.get_item_configuration(tag)["user_data"] for tag in dpg.get_item_children("dashboard", 1)]
        delta = len(mj_meta) - len(tags)
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
                        dpg.add_button(label="Reload Code", user_data=i, callback=self.reload_code_cb)
        positions = { m  : i for i, m in enumerate(reversed(sorted(meta, key=lambda i: mj_meta[i].absolute_completion()))) }
        for tag, m_id in zip(tags, meta):
            m = mj_meta[m_id]
            dpg.configure_item(tag.tag, label=f"Car #{m.id} - {m.label}")
            dpg.configure_item(tag.pending_id, show=not m.finished)
            dpg.configure_item(tag.finished_id, show=m.finished)
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
                dpg.set_value(tag.finished_text_id, f"Car finished {m.laps} laps in {sum(m.times):.2f} seconds! ({ordinal(self.mj.winners[m.id])} place)")

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
        if len(self.mj.meta) > 0:
            self.mj.watching = ((self.mj.watching or 0) + 1) % len(self.mj.meta)

    def focus_on_previous_car(self):
        """
        Locks the camera to the previous car in the sequence. Cycles at the end
        """
        if len(self.mj.meta) > 0:
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
            self.inject_meta([meta.id for meta in self.mj.meta])
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
            
        def new_car_cb(sender, value):
            tags = Tags()
            with dpg.group(parent="cars", user_data=tags) as group:
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_input_text(tag=tags.name_id, label="Driver Name")
                        dpg.add_input_text(tag=tags.path_id, label="Driver Path")
                        dpg.add_input_text(tag=tags.icon_id, label="Icon", callback=icon_cb,
                                           user_data=[tags.texture_id, tags.image_id])
                        dpg.add_color_edit(tag=tags.primary_id, label="Primary")
                        dpg.add_color_edit(tag=tags.secondary_id, label="Secondary")
                    with dpg.group(tag=tags.icon_group_id):
                        pass
                dpg.add_button(label="Delete", callback=delete_car_cb, user_data=group, width=-1)
                dpg.add_separator()
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
        dpg.add_texture_registry(tag="car icons")
        with dpg.window(tag="cars modal", width=400, height=400, no_scrollbar=True):
            with dpg.table(header_row=False):
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_button(label="New Car", callback=new_car_cb, width=-1)
                    dpg.add_button(label="Clear", callback=clear, width=-1)
                    dpg.add_button(label="Load from Sim", callback=lambda: load(self.mj.cars), width=-1)
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
        if self.mj.viewer is None:
            # inline renderer needs to be restarted when the window
            # resizes
            self.mj.start_inline_render_thread()
        # print("FINISH")

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
        self.camera_friction = [0.01, 0.01]
        self.watching = None
        self.rendered_dir = "rendered"
        self.template_dir = "template"
        self._camera = mujoco.MjvCamera()
        self.options = {}
        self.cars = []
        self.track = track
        
        try:
            with open("aigp_settings.json") as data_file:
                data = json.load(data_file)
        except FileNotFoundError as e:
            data = {}
            print("Not loading settings from file")

        self.declare("lock_camera", False, label="Lock Camera", data=data,
                     description="If this option is set, the camera angle will be kept in line with the angle of the vehicle being watched")
        self.declare("manual_control", False, label="Manual Control", persist=False,
                     description="Control the car being watched with the W, A, S and D keys")
        self.declare("manual_control_speed", 3.0, label="Manual Control Speed", data=data,
                     description="The speed at which the car will drive when being controlled manually")
        self.declare("cars_path", "cars.json", _type=str, label="Cars Path", persist=False)
        self.declare("lap_target", 10, data=data, label="Lap Target")
        self.declare("max_fps", 30, data=data, label="Max FPS")
        self.declare("cinematic_camera", False, data=data, label="Cinematic Camera")
        self.declare("pause_on_reload", True, data=data, label="Pause on Reload")
        self.declare("save_on_exit", True, data=data, label="Save on Exit", present=False,
                     description="Save to `aigp_settings.json` on exit")
        self.declare("max_geom", 1500, data=data, label="Mujoco Geom Limit", persist=False,
                     description="The number of entities that mujoco will render")
        self.declare("rangefinder_alpha", 0.1, label="Rangefinder Intensity", data=data, callback=self.rangefinder)

        self.meta = []
        self.shadows = {}
        self.steps = 0

        self.viewer = None
        self.kill_inline_render_event = Event()
        self.render_finished = Event()
        self.render_finished.set()

    @property
    def camera(self):
        if self.viewer is None or type(self.viewer) is str:
            return self._camera
        else:
            return self.viewer.cam

    def perturb_camera(self, dx, dy):
        if not self.option("cinematic_camera"):
            self.camera.azimuth   += dx
            self.camera.elevation += dy
        else:
            self.camera_vel[0] += dx / 100
            self.camera_vel[1] += dy / 100

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
        for m in self.meta:
            self.unshadow(m.id)
        metas = []
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
            driver = runtime_import(path).Driver()
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
        if self.watching is not None and self.watching >= len(self.meta):
            self.watching = None
        self.shadows = {}
        self.steps = 0
        self.winners = {}
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
        chunk(os.path.join(self.template_dir, f"{self.track}.png"), verbose=False, force=True)
        if self.option("cars_path") is not None:
            cars_path = os.path.join(self.template_dir, "cars", self.option("cars_path"))
            try:
                with open(cars_path) as cars_file:
                    self.cars = json.load(cars_file)
            except Exception:
                print(f"ERROR: Could not read from `{cars_path}`")
                self.cars = []
        produce_mjcf(rangefinders=90, cars=self.cars) # tweak cars_path here once you have multiple configs
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
        self.restart_render_thread()
        self.reload()

    def physics(self):
        Thread(target=self.physics_thread).start()

    def restart_render_thread(self):
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
            if self.option("lock_camera") and self.watching is not None:
                quat = self.meta[self.watching].joint.qpos[3:]
                desired_azimuth = quaternion_to_angle(*quat)
                camera_azimuth = self.camera.azimuth % 360
                desired_azimuth = math.degrees(desired_azimuth) % 360
                delta = (desired_azimuth - camera_azimuth + 180) % 360 - 180
                self.perturb_camera(delta * 0.01, 0)
            if self.option("cinematic_camera"):
                self.camera_vel[0] *= (1 - self.camera_friction[0])
                self.camera_vel[1] *= (1 - self.camera_friction[1])
                self.camera.azimuth += self.camera_vel[0]
                self.camera.elevation += self.camera_vel[1]
            if self.launch_viewer_event.is_set():
                if self.viewer is not None:
                    self.viewer.close()
                self.viewer = "John the Baptist"
                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback = lambda keycode: self.mv.release_key_cb(None, keycode)
                )
                self.launch_viewer_event.clear()
                self.kill_inline_render_event.set()
                self.mv.viewport_resize_event.set()
            if self.kill_viewer_event.is_set():
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                self.kill_viewer_event.clear()
                self.start_inline_render_thread()
                self.mv.viewport_resize_event.set()
            if self.watching is not None:
                self.camera.lookat[:] = self.data.body(f"car #{self.watching}").xpos[:]
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
                if not meta.off_track:
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
                
                if self.option("manual_control") and self.watching == meta.id:
                    speed, steering_angle = self.mv.speed, self.mv.steering_angle
                else:
                    # TODO: run this in its own thread so that the user can do things like time.sleep() without
                    # blocking the executor
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
        if width * height == 0:
            print("Window area is zero, not going to render anything")
            self.render_finished.set()
            self.kill_inline_render_event.clear()
            return
        self.renderer = Renderer(self.model, width=width, height=height, max_geom=self.option("max_geom"))
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

    
