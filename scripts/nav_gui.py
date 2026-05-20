import sys
import os
import time
import tkinter as tk
from tkinter import ttk
import math

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Rover import Rover
from Plant import Plant
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Assigned in setup_environment() before any detect_* call
client = None
sim    = None


# ── Environment setup ──────────────────────────────────────────────────────

def setup_environment():
    _client = RemoteAPIClient()
    _sim    = _client.getObject('sim')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir   = os.path.dirname(script_dir)

    rover_path  = os.path.join(base_dir, "models", "rover2.ttm")
    wiatka_path = os.path.join(base_dir, "models", "big_wiatka.ttm")
    # prefer plant1.ttm (has aruco_plane); fall back to plant.ttm
    plant_path = os.path.join(base_dir, "models", "plant1.ttm")
    if not os.path.exists(plant_path):
        plant_path = os.path.join(base_dir, "models", "plant.ttm")

    # ── Stop and wait ──────────────────────────────────────────────────────
    _sim.stopSimulation()
    while _sim.getSimulationState() != _sim.simulation_stopped:
        time.sleep(0.1)

    # ── Clean slate: remove all managed objects by prefix ─────────────────
    # (same approach as test_rover.py — removeModel first to clear full tree)
    CLEANUP = ('rover', 'base_cube', 'charging_station', 'ground_plane',
               'plant_', 'big_wiatka', 'terrain')
    for obj in _sim.getObjectsInTree(_sim.handle_scene, _sim.handle_all, 1):
        if _sim.getObjectParent(obj) != -1:
            continue
        alias = _sim.getObjectAlias(obj, 0)
        if any(alias.startswith(p) for p in CLEANUP):
            try:
                _sim.removeModel(obj)
            except Exception:
                try:
                    _sim.removeObject(obj)
                except Exception:
                    pass

    # ── Layout constants ───────────────────────────────────────────────────
    N_ROVERS      = 7
    DISTANCE_STEP = 7
    WIATKA_X      = 10.0
    WIATKA_Y      = -30.0
    SAFE_MARGIN   = 20

    # Charging stations in a 2×2 grid under the bower
    CHARGING_STATIONS = {
        1: (WIATKA_X - 1.5, WIATKA_Y + 1.5, 0.04),
        2: (WIATKA_X + 1.5, WIATKA_Y + 1.5, 0.04),
        3: (WIATKA_X - 1.5, WIATKA_Y - 1.5, 0.04),
        4: (WIATKA_X + 1.5, WIATKA_Y - 1.5, 0.04),
    }

    half = N_ROVERS // 2
    rover_positions = []
    for i in range(N_ROVERS):
        x = (WIATKA_X - (half - i - 1) * DISTANCE_STEP - SAFE_MARGIN
             if i < half
             else WIATKA_X + (i - half) * DISTANCE_STEP + SAFE_MARGIN)
        rover_positions.append([x, -30.0, 2.0])   # z=2.0 clears any terrain bumps
    rover_names = [f"rover_{i+1}" for i in range(N_ROVERS)]

    ARM_JOINTS = ['arm_base_joint', 'shoulder_joint', 'elbow_joint',
                  'wrist_joint', 'gripper_left_joint', 'gripper_right_joint']

    PLANTS_M, PLANTS_N = 5, 4
    plant_positions, plant_names = [], []
    for m in range(PLANTS_M):
        for n in range(PLANTS_N):
            plant_positions.append([-30 + m * 20, 0 + n * 20, 2.0])
            plant_names.append(f"plant_{m}_{n}")

    # ── Uneven terrain (heightfield) ───────────────────────────────────────
    PTS         = 64
    TERRAIN_SZ  = 200.0
    heights = []
    for yi in range(PTS):
        for xi in range(PTS):
            fx = xi / (PTS - 1) * 8 * math.pi
            fy = yi / (PTS - 1) * 6 * math.pi
            h = (math.sin(fx * 0.5) * math.cos(fy * 0.4) * 0.30 +
                 math.sin(fx * 1.3 + 0.7) * math.sin(fy * 1.1) * 0.15 +
                 math.cos(fx * 2.5) * math.sin(fy * 2.0 + 1.2) * 0.10)
            heights.append(h)   # ±0.55 m variation

    terrain_h = _sim.createHeightfieldShape(0, 40, PTS, PTS, TERRAIN_SZ, heights)
    _sim.setObjectPosition(terrain_h, -1, [0, 0, 0])
    _sim.setObjectAlias(terrain_h, 'ground_plane')
    _sim.setShapeColor(terrain_h, None,
                       _sim.colorcomponent_ambient_diffuse, [0.45, 0.30, 0.22])
    _sim.setObjectInt32Param(terrain_h, _sim.shapeintparam_static, 1)
    _sim.setObjectInt32Param(terrain_h, _sim.shapeintparam_respondable, 1)
    print("Stworzono teren")

    # ── Charging station pads ──────────────────────────────────────────────
    for sid, (x, y, z) in CHARGING_STATIONS.items():
        pad = _sim.createPrimitiveShape(
            _sim.primitiveshape_cylinder, [1.0, 1.0, 0.08], 0)
        _sim.setObjectPosition(pad, -1, [x, y, 0.04])
        _sim.setObjectAlias(pad, f'charging_station_{sid}')
        _sim.setShapeColor(pad, None, _sim.colorcomponent_ambient_diffuse, [1.0, 0.75, 0.0])
        _sim.setObjectInt32Param(pad, _sim.shapeintparam_static, 1)
        _sim.setObjectInt32Param(pad, _sim.shapeintparam_respondable, 0)

    # ── Rovers ────────────────────────────────────────────────────────────
    rover_jmaps = {}
    if os.path.exists(rover_path):
        base_rv = _sim.loadModel(rover_path)
        rv_handles = [base_rv]
        for _ in range(N_ROVERS - 1):
            rv_handles.append(_sim.copyPasteObjects([base_rv], 1)[0])
        for h, pos, name in zip(rv_handles, rover_positions, rover_names):
            _sim.setObjectPosition(h, -1, pos)
            _sim.setObjectOrientation(h, -1, [0, 0, math.radians(90)])
            _sim.setObjectAlias(h, name)
            joints = _sim.getObjectsInTree(h, _sim.object_joint_type, 0)
            rover_jmaps[name] = {_sim.getObjectAlias(j, 0): j for j in joints}
        print(f"Spawned {N_ROVERS} rovers")
    else:
        print(f"BLAD: brak {rover_path}")

    # ── Wiatka (shelter) ──────────────────────────────────────────────────
    if os.path.exists(wiatka_path):
        wh = _sim.loadModel(wiatka_path)
        _sim.setObjectPosition(wh, -1, [WIATKA_X, WIATKA_Y, 5.0])
        _sim.setObjectAlias(wh, 'big_wiatka_1')
        print("Spawned big_wiatka_1")

    # ── Plants with ArUco markers ──────────────────────────────────────────
    if os.path.exists(plant_path):
        base_ph   = _sim.loadModel(plant_path)
        p_handles = [base_ph]
        for _ in range(len(plant_positions) - 1):
            p_handles.append(_sim.copyPasteObjects([base_ph], 1)[0])

        use_aruco = 'plant1' in os.path.basename(plant_path)
        if use_aruco:
            try:
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            except AttributeError:
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

        print(f"Generowanie {len(plant_positions)} roslin...")
        for i, (h, pos, name) in enumerate(zip(p_handles, plant_positions, plant_names)):
            _sim.setObjectPosition(h, -1, pos)
            _sim.setObjectOrientation(h, -1, [0, 0, math.radians(-90)])
            _sim.setObjectAlias(h, name)
            if use_aruco:
                try:
                    plane_h     = _sim.getObject('./aruco_plane', {'proxy': h})
                    tex_res     = 512
                    marker_size = 392
                    visual_id   = i + 10
                    try:
                        marker = cv2.aruco.generateImageMarker(aruco_dict, visual_id, marker_size)
                    except AttributeError:
                        marker = cv2.aruco.drawMarker(aruco_dict, visual_id, marker_size)
                    img = np.ones((tex_res, tex_res), dtype=np.uint8) * 255
                    off = (tex_res - marker_size) // 2
                    img[off:off+marker_size, off:off+marker_size] = marker
                    rgb = cv2.flip(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 0)
                    tmp, tid, _ = _sim.createTexture(
                        "", 0, [0.15, 0.15], [1, 1], [0, 0, 0], 1, [tex_res, tex_res])
                    _sim.setShapeTexture(plane_h, tid, 0, 0, [0.3, 0.3])
                    _sim.removeObject(tmp)
                    _sim.writeTexture(tid, 0, rgb.tobytes(), 0, 0, tex_res, tex_res)
                    _sim.setObjectSizeValues(plane_h, [0.3, 0.3, 0.001])
                    print(f"  {name} aruco id={visual_id}")
                except Exception as e:
                    print(f"  ArUco {name}: {e}")
    else:
        print(f"BLAD: brak modelu rosliny: {plant_path}")

    # ── Start simulation ───────────────────────────────────────────────────
    _sim.startSimulation()
    time.sleep(1.0)

    # Reset arm joints (like test_rover.py) so the arm doesn't flail
    for name, jmap in rover_jmaps.items():
        for jname in ARM_JOINTS:
            if jname in jmap:
                _sim.setJointMode(jmap[jname], _sim.jointmode_force, 0)
                _sim.setJointTargetPosition(jmap[jname], 0)
    time.sleep(0.5)

    return _client, _sim

# ── Scene discovery — direct alias lookups (no full-scene scan) ────────────

def detect_base():
    try:
        with Rover._class_lock:
            h = sim.getObject('/base_cube')
            return sim.getObjectPosition(h, -1)
    except Exception:
        return None

def detect_charging_stations():
    stations = {}
    for sid in range(1, 5):
        try:
            with Rover._class_lock:
                h = sim.getObject(f'/charging_station_{sid}')
                stations[sid] = tuple(sim.getObjectPosition(h, -1))
        except Exception:
            pass
    return stations

def detect_rovers(charging_stations: dict, base_pos=None) -> list:
    _fallback = {1:(0,0,0), 2:(0,0,0), 3:(0,0,0), 4:(0,0,0)}
    cs = charging_stations or _fallback

    rover_data = []
    for i in range(1, 21):
        try:
            with Rover._class_lock:
                obj    = sim.getObject(f'/rover_{i}')
                joints = sim.getObjectsInTree(obj, sim.object_joint_type, 0)
                jnames = {sim.getObjectAlias(j, 0) for j in joints}
                if 'front_left_wheel_joint' not in jnames:
                    continue
                pos = sim.getObjectPosition(obj, -1)
            rover_data.append((obj, f'rover_{i}', pos))
        except Exception:
            continue

    if not rover_data:
        # Fallback: full-scene scan (slower, keeps compatibility)
        with Rover._class_lock:
            for obj in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 1):
                if sim.getObjectParent(obj) != -1:
                    continue
                alias = sim.getObjectAlias(obj, 0)
                if not alias.startswith('rover'):
                    continue
                joints = sim.getObjectsInTree(obj, sim.object_joint_type, 0)
                jnames = {sim.getObjectAlias(j, 0) for j in joints}
                if 'front_left_wheel_joint' not in jnames:
                    continue
                pos = sim.getObjectPosition(obj, -1)
                rover_data.append((obj, alias, pos))

    rovers = []
    for obj, name, pos in rover_data:
        rv = Rover(sim=sim, handle=obj, spawn_coords=pos,
                   charging_stations=cs, base_pos=base_pos)
        rovers.append(rv)
        print(f"Wykryto rover: {name}")

    rovers.sort(key=lambda r: r.name)
    for rv in rovers:
        rv.all_rovers = rovers
    return rovers

def detect_plants() -> list:
    plants = []
    for r in range(20):
        found_in_row = False
        for c in range(20):
            alias = f"plant_{r}_{c}"
            try:
                with Rover._class_lock:
                    handle = sim.getObject(f'/{alias}')
                p = Plant(sim=sim, handle=handle, name=alias)
                plants.append(p)
                found_in_row = True
                print(f"Wykryto rosliny: {alias}")
            except Exception:
                break
        if not found_in_row:
            break
    plants.sort(key=lambda p: p.name)
    return plants


# ── GUI constants ──────────────────────────────────────────────────────────
CANVAS_SIZE  = 500
DRAG_THRESH  = 15
DEFAULT_VIEW = (-60.0, 60.0, -60.0, 60.0)

ROVER_COLORS = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
]
STATUS_COLORS = {
    "idle":     "gray",
    "moving":   "#3498db",
    "arrived":  "#2ecc71",
    "blocked":  "#e74c3c",
    "charging": "#f1c40f",
    "solar":    "#a8e063",
    "dead":     "#7f8c8d",
}


# ── Main application ───────────────────────────────────────────────────────
class NavApp:
    def __init__(self, root, rovers, base_pos, stations, plants=None):
        self.root     = root
        self.rovers   = rovers
        self.stations = stations
        self.base_pos = base_pos
        self.plants   = plants or []

        self._drag_start      = None
        self._pending_heading = None
        self._target_items    = []

        # plant_scan[name] = {state, received, vals(temp), display(persistent)}
        self._plant_scan = {}

        self._view              = list(DEFAULT_VIEW)
        self.rover_target_items = {}

        self._tooltip_win = None
        self._tooltip_lbl = None

        root.title("Nawigacja Roverow")
        self._build_ui()
        self._update_loop()

    # ── Coordinate helpers ─────────────────────────────────────────────────

    def _w2c(self, wx, wy):
        xmin, xmax, ymin, ymax = self._view
        return ((wx - xmin) / (xmax - xmin) * CANVAS_SIZE,
                (1 - (wy - ymin) / (ymax - ymin)) * CANVAS_SIZE)

    def _c2w(self, cx, cy):
        xmin, xmax, ymin, ymax = self._view
        return (cx / CANVAS_SIZE * (xmax - xmin) + xmin,
                (1 - cy / CANVAS_SIZE) * (ymax - ymin) + ymin)

    # ── Zoom ───────────────────────────────────────────────────────────────

    def _on_zoom(self, event):
        wx, wy  = self._c2w(event.x, event.y)
        zoom_in = (getattr(event, 'delta', 0) > 0) or (event.num == 4)
        factor  = 0.85 if zoom_in else 1.15
        xmin, xmax, ymin, ymax = self._view
        self._view = [wx + (xmin - wx) * factor, wx + (xmax - wx) * factor,
                      wy + (ymin - wy) * factor, wy + (ymax - wy) * factor]
        self._redraw_static()

    def _reset_zoom(self):
        self._view = list(DEFAULT_VIEW)
        self._redraw_static()

    def _redraw_static(self):
        self.canvas.delete("static")
        for item in self.rover_dots.values():
            self.canvas.delete(item)
        for item in self.rover_headings.values():
            self.canvas.delete(item)
        self.rover_dots.clear()
        self.rover_headings.clear()
        for items in self.rover_target_items.values():
            for i in items:
                self.canvas.delete(i)
        self.rover_target_items.clear()
        self._draw_grid()
        self._draw_base_and_stations()
        self._draw_plant_dots()

    # ── UI builder ─────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        top  = tk.Frame(root)
        top.pack(padx=10, pady=10, fill="both", expand=True)

        map_frame = tk.LabelFrame(
            top, text="Mapa  (klik=cel | drag=cel+kat | scroll=zoom)")
        map_frame.pack(side=tk.LEFT, padx=(0, 8))

        self.canvas = tk.Canvas(map_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#1a1a2e", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-4>",        self._on_zoom)
        self.canvas.bind("<Button-5>",        self._on_zoom)
        self.canvas.bind("<MouseWheel>",      self._on_zoom)
        self.canvas.bind("<Motion>",          self._on_canvas_motion)
        self.canvas.bind("<Leave>",           lambda e: self._hide_tooltip())

        self.rover_dots      = {}
        self.rover_headings  = {}
        self.station_ovals   = {}
        self.plant_map_items = {}

        self._draw_grid()
        self._draw_base_and_stations()
        self._draw_plant_dots()

        # Rover list
        list_frame = tk.LabelFrame(top, text="Rovery")
        list_frame.pack(side=tk.LEFT, fill="y")

        self.rover_rows = {}
        for i, rv in enumerate(self.rovers):
            color = ROVER_COLORS[i % len(ROVER_COLORS)]
            row   = tk.Frame(list_frame)
            row.pack(fill="x", padx=6, pady=4)

            tk.Label(row, text="●", fg=color, font=("Arial", 14)).pack(side=tk.LEFT)
            tk.Label(row, text=rv.name, width=9, anchor="w").pack(side=tk.LEFT)

            status_lbl = tk.Label(row, text="idle", width=9, anchor="w", fg="gray")
            status_lbl.pack(side=tk.LEFT)

            batt_lbl = tk.Label(row, text="100%", width=5, anchor="e",
                                fg="#2ecc71", font=("Courier", 9, "bold"))
            batt_lbl.pack(side=tk.LEFT, padx=(0, 4))

            tk.Button(row, text="Spawn", width=5,
                      command=lambda r=rv: r.go_to_base()).pack(side=tk.LEFT, padx=2)

            charge_btn = tk.Button(row, text="Charge", width=6,
                                   command=lambda r=rv: self._go_charge(r))
            charge_btn.pack(side=tk.LEFT, padx=2)

            panels_btn = tk.Button(row, text="Open", width=6,
                                   command=lambda r=rv: self._toggle_panels(r))
            panels_btn.pack(side=tk.LEFT, padx=2)

            tk.Button(row, text="Stop", width=4,
                      command=lambda r=rv: r.stop()).pack(side=tk.LEFT, padx=2)

            self.rover_rows[rv.name] = (color, status_lbl, batt_lbl, charge_btn, panels_btn)

        if self.plants:
            self._build_plant_panel(top)

        # Send-goal strip
        ctrl = tk.LabelFrame(root, text="Wyslij cel", padx=8, pady=8)
        ctrl.pack(padx=10, pady=(0, 10), fill="x")

        tk.Label(ctrl, text="X:").grid(row=0, column=0, padx=4)
        self.x_var = tk.StringVar(value="0")
        tk.Entry(ctrl, textvariable=self.x_var, width=8).grid(row=0, column=1, padx=4)

        tk.Label(ctrl, text="Y:").grid(row=0, column=2, padx=4)
        self.y_var = tk.StringVar(value="0")
        tk.Entry(ctrl, textvariable=self.y_var, width=8).grid(row=0, column=3, padx=4)

        tk.Label(ctrl, text="Kat:").grid(row=0, column=4, padx=4)
        self.heading_var = tk.StringVar(value="-")
        tk.Label(ctrl, textvariable=self.heading_var, width=7,
                 anchor="w").grid(row=0, column=5, padx=4)

        tk.Label(ctrl, text="Rover:").grid(row=0, column=6, padx=4)
        rover_choices = ["Wszystkie"] + [r.name for r in self.rovers]
        self.target_rover_var = tk.StringVar(value="Wszystkie")
        ttk.Combobox(ctrl, textvariable=self.target_rover_var,
                     values=rover_choices, width=12,
                     state="readonly").grid(row=0, column=7, padx=4)

        tk.Button(ctrl, text="Wyslij", width=10, bg="#2ecc71",
                  command=self._send).grid(row=0, column=8, padx=8)
        tk.Button(ctrl, text="Stop wszystko", width=14, bg="#e74c3c", fg="white",
                  command=self._stop_all).grid(row=0, column=9, padx=4)
        tk.Button(ctrl, text="Reset zoom", width=10,
                  command=self._reset_zoom).grid(row=0, column=10, padx=4)

        self.status_lbl = tk.Label(root, text="", anchor="w", fg="gray")
        self.status_lbl.pack(padx=10, fill="x")

    # ── Static map drawing ─────────────────────────────────────────────────

    def _draw_grid(self):
        xmin, xmax, ymin, ymax = self._view
        step = next((s for s in (1,2,5,10,20,50,100)
                     if (xmax - xmin) / s <= 20), 100)
        import math as _m
        x = _m.ceil(xmin / step) * step
        while x <= xmax:
            cx, _ = self._w2c(x, 0)
            self.canvas.create_line(cx, 0, cx, CANVAS_SIZE, fill="#2a2a4a", tags="static")
            x += step
        y = _m.ceil(ymin / step) * step
        while y <= ymax:
            _, cy = self._w2c(0, y)
            self.canvas.create_line(0, cy, CANVAS_SIZE, cy, fill="#2a2a4a", tags="static")
            y += step
        ox, oy = self._w2c(0, 0)
        self.canvas.create_line(ox, 0, ox, CANVAS_SIZE, fill="#3a3a6a", width=2, tags="static")
        self.canvas.create_line(0, oy, CANVAS_SIZE, oy, fill="#3a3a6a", width=2, tags="static")

    def _draw_base_and_stations(self):
        self.station_ovals = {}
        if self.base_pos:
            bx, by = self.base_pos[0], self.base_pos[1]
            x1, y1 = self._w2c(bx - 1.5, by + 1.5)
            x2, y2 = self._w2c(bx + 1.5, by - 1.5)
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         fill="#1a3570", outline="#4466dd",
                                         width=2, tags="static")
            cx, cy = self._w2c(bx, by)
            self.canvas.create_text(cx, cy, text="BASE", fill="white",
                                    font=("Arial", 7, "bold"), tags="static")

        for sid, (sx, sy, *_) in self.stations.items():
            cx, cy = self._w2c(sx, sy)
            if self.base_pos:
                bx, by = self._w2c(self.base_pos[0], self.base_pos[1])
                self.canvas.create_line(bx, by, cx, cy, fill="#444466",
                                        width=1, dash=(3,4), tags="static")
            r = 7
            oval = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                           outline="#f1c40f", fill="",
                                           width=2, tags="static")
            self.canvas.create_text(cx, cy, text=str(sid), fill="#f1c40f",
                                    font=("Arial", 7, "bold"), tags="static")
            self.station_ovals[sid] = oval

    def _draw_plant_dots(self):
        for dot, lbl in self.plant_map_items.values():
            self.canvas.delete(dot)
            self.canvas.delete(lbl)
        self.plant_map_items = {}
        for p in self.plants:
            cx, cy = self._w2c(p.pos[0], p.pos[1])
            tag = f"ptag_{p.name}"
            dot = self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5,
                                          fill="#27ae60", outline="#1e8449",
                                          width=1, tags=(tag, "static"))
            short = p.name.replace('plant_', 'P').replace('_', '')
            lbl   = self.canvas.create_text(cx, cy-10, text=short,
                                            fill="#a8e063", font=("Arial", 6),
                                            tags=(tag, "static"))
            self.canvas.tag_bind(tag, "<Button-1>",
                                 lambda e, pl=p: self._send_to_plant(pl))
            self.plant_map_items[p.name] = (dot, lbl)

    # ── Plant panel ────────────────────────────────────────────────────────

    def _build_plant_panel(self, parent):
        plants_lf = tk.LabelFrame(parent, text="Rosliny")
        plants_lf.pack(side=tk.LEFT, fill="y", padx=(4, 0))

        canv = tk.Canvas(plants_lf, width=420, height=CANVAS_SIZE,
                         bg="white", highlightthickness=0)
        vsb  = tk.Scrollbar(plants_lf, orient="vertical", command=canv.yview)
        canv.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill="y")
        canv.pack(side=tk.LEFT, fill="both")

        inner = tk.Frame(canv, bg="white")
        canv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canv.configure(scrollregion=canv.bbox("all")))

        self.plant_widgets = {}

        for p in self.plants:
            self._plant_scan[p.name] = {
                'state':   'idle',
                'received': set(),
                'vals':     {},       # temp while scanning
                'display':  {},       # persistent: h/f/d last scanned values
            }

            row = tk.Frame(inner, relief="groove", bd=1, bg="white")
            row.pack(fill="x", padx=3, pady=2, ipady=3)

            tk.Label(row, text=p.name, width=10, anchor="w",
                     font=("Courier", 8, "bold"), bg="white").pack(side=tk.LEFT, padx=(4,2))

            val_labels = {}
            for key, letter in [("h", "H"), ("f", "F"), ("d", "D")]:
                tk.Label(row, text=f"{letter}:", font=("Arial", 8),
                         fg="#555", bg="white").pack(side=tk.LEFT, padx=(4, 0))
                vl = tk.Label(row, text="--", width=4,
                              font=("Courier", 8), fg="#888", bg="white", anchor="e")
                vl.pack(side=tk.LEFT, padx=(1, 2))
                val_labels[key] = vl

            move_btn = tk.Button(row, text="Move", width=4, font=("Arial", 7),
                                 command=lambda pl=p: self._send_to_plant(pl))
            move_btn.pack(side=tk.LEFT, padx=(4, 1))

            scan_btn = tk.Button(row, text="Scan", width=4, font=("Arial", 7),
                                 command=lambda pl=p: self._start_plant_scan(pl))
            scan_btn.pack(side=tk.LEFT, padx=1)

            repair_btn = tk.Button(row, text="Repair", width=5, font=("Arial", 7),
                                   command=lambda pl=p: self._repair_plant(pl))
            repair_btn.pack(side=tk.LEFT, padx=(1, 4))

            self.plant_widgets[p.name] = {
                'h_val': val_labels["h"], 'f_val': val_labels["f"], 'd_val': val_labels["d"],
                'scan_btn': scan_btn, 'repair_btn': repair_btn,
            }

    # ── Map interaction ─────────────────────────────────────────────────────

    def _on_press(self, event):
        self._drag_start = (event.x, event.y)
        wx, wy = self._c2w(event.x, event.y)
        self.x_var.set(f"{wx:.1f}")
        self.y_var.set(f"{wy:.1f}")
        self._pending_heading = None
        self.heading_var.set("-")
        self._clear_target_marker()

    def _on_drag(self, event):
        if not self._drag_start:
            return
        sx, sy = self._drag_start
        dx, dy = event.x - sx, event.y - sy
        if math.hypot(dx, dy) > DRAG_THRESH:
            self._draw_target_arrow(sx, sy, math.atan2(-dy, dx))

    def _on_release(self, event):
        if not self._drag_start:
            return
        sx, sy = self._drag_start
        dx, dy = event.x - sx, event.y - sy
        if math.hypot(dx, dy) > DRAG_THRESH:
            self._pending_heading = math.atan2(-dy, dx)
            self.heading_var.set(f"{math.degrees(self._pending_heading):.1f}°")
            self._draw_target_arrow(sx, sy, self._pending_heading)
        else:
            self._pending_heading = None
            self.heading_var.set("-")
            self._draw_target_circle(sx, sy)
        self._drag_start = None

    def _clear_target_marker(self):
        for item in self._target_items:
            self.canvas.delete(item)
        self._target_items.clear()

    def _draw_target_circle(self, cx, cy):
        self._clear_target_marker()
        r = 8
        self._target_items.append(
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    outline="#f1c40f", width=2, fill=""))

    def _draw_target_arrow(self, cx, cy, angle, length=28):
        self._clear_target_marker()
        r = 5
        self._target_items.append(
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#f1c40f", outline=""))
        ex = cx + math.cos(angle) * length
        ey = cy - math.sin(angle) * length
        self._target_items.append(
            self.canvas.create_line(cx, cy, ex, ey, fill="#f1c40f", width=2, arrow=tk.LAST))

    # ── Tooltip ────────────────────────────────────────────────────────────

    def _on_canvas_motion(self, event):
        for p in self.plants:
            cx, cy = self._w2c(p.pos[0], p.pos[1])
            if math.hypot(event.x - cx, event.y - cy) <= 10:
                action_line = (f"\nAction: {p.action} {int(p.action_progress * 100)}%"
                               if p.action else "")
                text = (f"{p.name}\n"
                        f"Humidity:     {p.humidity:.0f}%\n"
                        f"Fertility:    {p.fertility:.0f}%\n"
                        f"Crop density: {p.crop_density:.0f}%"
                        f"{action_line}")
                self._show_tooltip(text, event.x, event.y)
                return
        self._hide_tooltip()

    def _show_tooltip(self, text, canvas_x, canvas_y):
        if self._tooltip_win is None:
            self._tooltip_win = tk.Toplevel(self.root)
            self._tooltip_win.wm_overrideredirect(True)
            self._tooltip_lbl = tk.Label(
                self._tooltip_win, text="", bg="#ffffc0",
                relief="solid", bd=1, font=("Courier", 8),
                justify=tk.LEFT, padx=6, pady=4)
            self._tooltip_lbl.pack()
        self._tooltip_lbl.config(text=text)
        rx = self.canvas.winfo_rootx() + canvas_x + 14
        ry = self.canvas.winfo_rooty() + canvas_y + 14
        self._tooltip_win.wm_geometry(f"+{rx}+{ry}")
        self._tooltip_win.deiconify()

    def _hide_tooltip(self):
        if self._tooltip_win:
            self._tooltip_win.withdraw()

    # ── Control actions ─────────────────────────────────────────────────────

    def _send(self):
        try:
            tx = float(self.x_var.get())
            ty = float(self.y_var.get())
        except ValueError:
            self.status_lbl.config(text="Blad: podaj poprawne liczby", fg="red")
            return
        chosen  = self.target_rover_var.get()
        targets = self.rovers if chosen == "Wszystkie" else [
            r for r in self.rovers if r.name == chosen]
        for rv in targets:
            rv.go_to(tx, ty, self._pending_heading)
        hdg_str = (f"{math.degrees(self._pending_heading):.1f}°"
                   if self._pending_heading is not None else "dowolna")
        self.status_lbl.config(
            text=f"Cel ({tx:.1f}, {ty:.1f}) kat={hdg_str} -> "
                 f"{', '.join(r.name for r in targets)}", fg="#2ecc71")
        cx, cy = self._w2c(tx, ty)
        if self._pending_heading is not None:
            self._draw_target_arrow(cx, cy, self._pending_heading)
        else:
            self._draw_target_circle(cx, cy)

    def _stop_all(self):
        for rv in self.rovers:
            rv.stop()
        self.status_lbl.config(text="Wszystkie rovery zatrzymane", fg="orange")

    def _go_charge(self, rv):
        claimed = {r.charging_target for r in self.rovers if r.charging_target is not None}
        for sid in range(1, 5):
            if sid not in claimed:
                rv.go_to_charging_station(sid)
                return

    def _toggle_panels(self, rv):
        if rv.panels_open:
            rv.close_panels()
        else:
            rv.open_panels()

    def _send_to_plant(self, plant):
        chosen = self.target_rover_var.get()
        if chosen == "Wszystkie":
            self.status_lbl.config(
                text="Wybierz konkretny rover przed wyslaniem do rosliny", fg="orange")
            return
        for rv in self.rovers:
            if rv.name == chosen:
                rv.go_to(plant.pos[0], plant.pos[1])
                self.status_lbl.config(
                    text=f"{rv.name} -> {plant.name} ({plant.pos[0]:.1f}, {plant.pos[1]:.1f})",
                    fg="#2ecc71")
                return

    def _start_plant_scan(self, plant):
        if self._plant_scan[plant.name]['state'] != 'idle':
            return
        plant.measure_all()
        self._plant_scan[plant.name].update(
            {'state': 'scanning', 'received': set(), 'vals': {}})

    def _repair_plant(self, plant):
        vals   = {'humidity': plant.humidity,
                  'fertility': plant.fertility,
                  'crop_density': plant.crop_density}
        worst  = min(vals, key=vals.get)
        if worst == 'humidity':
            plant.water()
        elif worst == 'fertility':
            plant.fertilize()
        else:
            plant.sow()

    # ── Periodic GUI refresh ────────────────────────────────────────────────

    def _update_loop(self):
        claimed = {rv.charging_target for rv in self.rovers
                   if rv.charging_target is not None}

        for sid, oval in self.station_ovals.items():
            self.canvas.itemconfig(oval, fill="#f1c40f" if sid in claimed else "")

        for i, rv in enumerate(self.rovers):
            color, status_lbl, batt_lbl, charge_btn, panels_btn = self.rover_rows[rv.name]

            if rv.charging and rv.charging_target is not None:
                disp = "charging"
            elif rv.panels_open and rv.charging:
                disp = "solar"
            else:
                disp = rv.status
            status_lbl.config(text=disp, fg=STATUS_COLORS.get(disp, "white"))

            b     = rv.battery
            bc    = "#2ecc71" if b > 50 else "#f39c12" if b > 20 else "#e74c3c"
            arrow = "↑" if rv.charging else ""
            batt_lbl.config(text=f"{arrow}{b:.0f}%", fg=bc)

            panels_btn.config(text="Close" if rv.panels_open else "Open")

            all_taken = len(claimed) >= len(self.stations)
            charge_btn.config(
                state=tk.DISABLED if (all_taken or rv.charging_target is not None
                                      or rv.panels_open or rv.status == "dead")
                else tk.NORMAL)
            panels_btn.config(state=tk.DISABLED if rv.status == "dead" else tk.NORMAL)

            try:
                cx, cy = self._w2c(rv.pos[0], rv.pos[1])
                if rv.name in self.rover_dots:
                    self.canvas.delete(self.rover_dots[rv.name])
                    self.canvas.delete(self.rover_headings[rv.name])
                outline, ow = ("#e74c3c", 3) if rv.status == "blocked" else ("white", 1)
                self.rover_dots[rv.name] = self.canvas.create_oval(
                    cx-8, cy-8, cx+8, cy+8, fill=color, outline=outline, width=ow)
                hx = cx + math.cos(rv.heading) * 14
                hy = cy - math.sin(rv.heading) * 14
                self.rover_headings[rv.name] = self.canvas.create_line(
                    cx, cy, hx, hy, fill="white", width=2, arrow=tk.LAST)
                lbl_key = f"lbl_{rv.name}"
                if lbl_key not in self.rover_dots:
                    self.rover_dots[lbl_key] = self.canvas.create_text(
                        cx, cy-14, text=rv.name, fill=color, font=("Arial", 7))
                else:
                    self.canvas.coords(self.rover_dots[lbl_key], cx, cy-14)
                self._draw_rover_route(rv, color)
            except Exception:
                pass

        self._update_plants()
        self.root.after(150, self._update_loop)

    def _draw_rover_route(self, rv, color):
        for item in self.rover_target_items.get(rv.name, []):
            self.canvas.delete(item)
        route = rv.route
        if not route or rv.status not in ('moving', 'blocked'):
            self.rover_target_items[rv.name] = []
            return
        items = []
        flat  = []
        for wx, wy in [(rv.pos[0], rv.pos[1])] + route:
            cx, cy = self._w2c(wx, wy)
            flat.extend([cx, cy])
        if len(flat) >= 4:
            items.append(self.canvas.create_line(
                *flat, fill=color, dash=(5, 4), width=1))
        fx, fy = self._w2c(route[-1][0], route[-1][1])
        items.append(self.canvas.create_line(fx-6, fy-6, fx+6, fy+6, fill=color, width=2))
        items.append(self.canvas.create_line(fx-6, fy+6, fx+6, fy-6, fill=color, width=2))
        self.rover_target_items[rv.name] = items

    def _update_plants(self):
        if not self.plants:
            return
        for plant in self.plants:
            wgt  = self.plant_widgets[plant.name]
            scan = self._plant_scan[plant.name]

            # Collect scan results as they arrive
            if scan['state'] == 'scanning':
                for param, key in [('humidity','h'), ('fertility','f'), ('crop_density','d')]:
                    if key not in scan['received']:
                        state, val = plant.get_measure(param)
                        if state == 'ready':
                            scan['vals'][key] = val
                            scan['received'].add(key)
                if scan['received'] == {'h', 'f', 'd'}:
                    # Persist values and return to idle
                    scan['display'] = dict(scan['vals'])
                    scan['state']   = 'idle'
                    scan['received'].clear()
                    scan['vals'].clear()

            # Update value labels: scan snapshot only (or "?" while scanning)
            if scan['state'] == 'scanning':
                for k in ('h', 'f', 'd'):
                    wgt[f'{k}_val'].config(text="?", fg="#3498db")
            elif scan['display']:
                d = scan['display']
                for k, key in (('h','h'), ('f','f'), ('d','d')):
                    v = d[key]
                    c = "#2ecc71" if v > 60 else "#f39c12" if v > 30 else "#e74c3c"
                    wgt[f'{k}_val'].config(text=f"{v:.0f}%", fg=c)
            else:
                for k in ('h', 'f', 'd'):
                    wgt[f'{k}_val'].config(text="--", fg="#888")

            # Buttons
            nearby = plant.rover_nearby(self.rovers)
            busy   = plant.action is not None
            wgt['scan_btn'].config(
                state=tk.DISABLED if (scan['state'] != 'idle' or not nearby) else tk.NORMAL)
            wgt['repair_btn'].config(
                state=tk.DISABLED if busy else tk.NORMAL)

            # Map dot colour by worst live parameter
            worst = min(plant.humidity, plant.fertility, plant.crop_density)
            dot_color = ("#e74c3c" if worst < 30 else "#f39c12" if worst < 60 else "#27ae60")
            if plant.name in self.plant_map_items:
                self.canvas.itemconfig(self.plant_map_items[plant.name][0], fill=dot_color)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client, sim = setup_environment()

    base_pos = detect_base()
    stations = detect_charging_stations()
    if not stations:
        print("UWAGA: brak stacji ladowania")

    rovers = detect_rovers(stations, base_pos=base_pos)
    if not rovers:
        print("Nie znaleziono roverow.")
    else:
        print(f"Znaleziono {len(rovers)} rover(ow): {[r.name for r in rovers]}")
        if base_pos:
            print(f"Baza: {[round(v,1) for v in base_pos]}")

        plants = detect_plants()
        print(f"Znaleziono {len(plants)} roslin")

        root = tk.Tk()
        NavApp(root, rovers, base_pos, stations, plants)
        root.mainloop()
