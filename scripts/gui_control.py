from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import tkinter as tk
import os
import math
import time

# --- SETUP ---
client = RemoteAPIClient()
sim = client.getObject('sim')

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
rover_path = os.path.join(base_dir, "models", "rover2.ttm")

sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

all_objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 1)
for obj in all_objects:
    if sim.getObjectParent(obj) == -1 and sim.getObjectAlias(obj, 0).startswith('rover'):
        sim.removeModel(obj)

rover_handle = sim.loadModel(rover_path)
sim.setObjectPosition(rover_handle, -1, [0, 0, 1.5])
sim.setObjectOrientation(rover_handle, -1, [0, 0, math.radians(-90)])
sim.setObjectAlias(rover_handle, "rover_1")

all_joints = sim.getObjectsInTree(rover_handle, sim.object_joint_type, 0)
joint_map = {sim.getObjectAlias(j, 0): j for j in all_joints}

WHEEL_NAMES = {
    "front_left_wheel_joint":  "left",
    "rear_left_wheel_joint":   "left",
    "front_right_wheel_joint": "right",
    "rear_right_wheel_joint":  "right",
}
ARM_JOINTS  = ['arm_base_joint', 'shoulder_joint', 'elbow_joint',
               'wrist_joint', 'gripper_left_joint', 'gripper_right_joint']
PANEL_JOINTS = ['left_panel_joint', 'right_panel_joint']

left_wheels, right_wheels = [], []
for name, side in WHEEL_NAMES.items():
    if name in joint_map:
        h = joint_map[name]
        sim.setJointMode(h, sim.jointmode_force, 0)
        try:
            sim.setJointTargetForce(h, 100)
        except Exception:
            pass
        (left_wheels if side == "left" else right_wheels).append(h)

sim.startSimulation()
time.sleep(1.0)

for name in WHEEL_NAMES:
    if name in joint_map:
        sim.setJointMode(joint_map[name], sim.jointmode_force, 0)

for name in ARM_JOINTS + PANEL_JOINTS:
    if name in joint_map:
        h = joint_map[name]
        sim.setJointMode(h, sim.jointmode_force, 0)
        sim.setJointTargetPosition(h, 0)

time.sleep(1.5)


# --- CONTROLLER ---
class RoverController:
    def __init__(self, speed=2.0):
        self.speed = speed

    def set_velocity(self, left_vel, right_vel):
        for h in left_wheels:
            sim.setJointTargetVelocity(h, left_vel)
        for h in right_wheels:
            sim.setJointTargetVelocity(h, right_vel)

    def forward(self):   self.set_velocity(-self.speed, -self.speed)
    def backward(self):  self.set_velocity( self.speed,  self.speed)
    def turn_left(self): self.set_velocity( self.speed, -self.speed)
    def turn_right(self):self.set_velocity(-self.speed,  self.speed)
    def stop(self):      self.set_velocity(0, 0)


rover = RoverController()


# --- GUI ---
root = tk.Tk()
root.title("Rover Control")
root.resizable(False, False)

# ── Jazda ──────────────────────────────────────────────────────────────────
drive_frame = tk.LabelFrame(root, text="Jazda", padx=10, pady=8)
drive_frame.pack(padx=12, pady=(10, 4), fill="x")

def drive_btn(parent, text, row, col, action, bg="#ddeedd"):
    btn = tk.Button(parent, text=text, width=5, height=2, bg=bg, font=("Arial", 14))
    btn.grid(row=row, column=col, padx=3, pady=3)
    btn.bind("<ButtonPress-1>",   lambda e: action())
    btn.bind("<ButtonRelease-1>", lambda e: rover.stop())

drive_btn(drive_frame, "↑", 0, 1, rover.forward)
drive_btn(drive_frame, "←", 1, 0, rover.turn_left)
drive_btn(drive_frame, "→", 1, 2, rover.turn_right)
drive_btn(drive_frame, "↓", 2, 1, rover.backward)
tk.Button(drive_frame, text="■", width=5, height=2,
          bg="#ff6b6b", font=("Arial", 14),
          command=rover.stop).grid(row=1, column=1, padx=3, pady=3)

speed_row = tk.Frame(drive_frame)
speed_row.grid(row=3, column=0, columnspan=3, pady=(8, 0))
tk.Label(speed_row, text="Prędkość:").pack(side=tk.LEFT)
speed_var = tk.DoubleVar(value=2.0)
tk.Scale(speed_row, from_=0.5, to=10.0, resolution=0.5,
         orient=tk.HORIZONTAL, length=180, variable=speed_var,
         command=lambda v: setattr(rover, 'speed', float(v))).pack(side=tk.LEFT)

# Klawiatura WASD
def on_key_press(e):
    k = e.keysym.lower()
    if k == 'w':     rover.forward()
    elif k == 's':   rover.backward()
    elif k == 'a':   rover.turn_left()
    elif k == 'd':   rover.turn_right()
    elif k == 'space': rover.stop()

def on_key_release(e):
    if e.keysym.lower() in ('w', 's', 'a', 'd'):
        rover.stop()

root.bind("<KeyPress>",   on_key_press)
root.bind("<KeyRelease>", on_key_release)

# ── Ramię ──────────────────────────────────────────────────────────────────
arm_frame = tk.LabelFrame(root, text="Ramię", padx=10, pady=8)
arm_frame.pack(padx=12, pady=4, fill="x")

ARM_CONFIG = [
    ('arm_base_joint',      'Baza',        -3.14,  3.14),
    ('shoulder_joint',      'Ramię',       -1.57,  1.57),
    ('elbow_joint',         'Łokieć',      -1.57,  1.57),
    ('wrist_joint',         'Nadgarstek',  -1.57,  1.57),
    ('gripper_left_joint',  'Chwytak L',   -0.5,   0.5),
    ('gripper_right_joint', 'Chwytak P',   -0.5,   0.5),
]

arm_sliders = {}

for i, (name, label, lo, hi) in enumerate(ARM_CONFIG):
    if name not in joint_map:
        continue
    tk.Label(arm_frame, text=label, width=12, anchor='w').grid(row=i, column=0, sticky='w')

    val_lbl = tk.Label(arm_frame, text=" 0.00", width=6, anchor='w')
    val_lbl.grid(row=i, column=2, padx=(4, 0))

    def make_cmd(n, lbl):
        def cmd(v):
            sim.setJointTargetPosition(joint_map[n], float(v))
            lbl.config(text=f"{float(v):+.2f}")
        return cmd

    s = tk.Scale(arm_frame, from_=lo, to=hi, resolution=0.01,
                 orient=tk.HORIZONTAL, length=240,
                 command=make_cmd(name, val_lbl), showvalue=False)
    s.set(0)
    s.grid(row=i, column=1, padx=5, pady=1)
    arm_sliders[name] = s

def reset_arm():
    for name, s in arm_sliders.items():
        s.set(0)
        sim.setJointTargetPosition(joint_map[name], 0)

tk.Button(arm_frame, text="Reset do zera", command=reset_arm).grid(
    row=len(ARM_CONFIG), column=0, columnspan=3, pady=(8, 0))

# ── Panele słoneczne ───────────────────────────────────────────────────────
panel_frame = tk.LabelFrame(root, text="Panele słoneczne", padx=10, pady=8)
panel_frame.pack(padx=12, pady=(4, 10), fill="x")

def set_panels(left_angle, right_angle):
    if 'left_panel_joint' in joint_map:
        sim.setJointTargetPosition(joint_map['left_panel_joint'], left_angle)
    if 'right_panel_joint' in joint_map:
        sim.setJointTargetPosition(joint_map['right_panel_joint'], right_angle)

tk.Button(panel_frame, text="☀  Otwórz", width=14, height=2,
          bg="#90EE90", command=lambda: set_panels(0, 0)).pack(side=tk.LEFT, padx=8)
tk.Button(panel_frame, text="✕  Zamknij", width=14, height=2,
          bg="#FFB6C1", command=lambda: set_panels(-math.pi / 2, math.pi / 2)).pack(side=tk.LEFT, padx=8)

root.mainloop()
