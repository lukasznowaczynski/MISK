import sys
import os
import math
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Rover import Rover
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir   = os.path.dirname(script_dir)
rover_path = os.path.join(base_dir, "models", "rover2.ttm")
plant_path = os.path.join(base_dir, "models", "plant.ttm")

# ── Base cube and charging station layout ─────────────────────────────────
BASE_POS    = [20.0, 0.0]
BASE_SIZE   = 3.0
_OFFSET     = BASE_SIZE / 2 + 2.5   # 4.0 m from centre to each station pad

CHARGING_STATIONS = {
    1: (BASE_POS[0] + _OFFSET, BASE_POS[1],           0.5),  # +X face
    2: (BASE_POS[0] - _OFFSET, BASE_POS[1],           0.5),  # -X face
    3: (BASE_POS[0],            BASE_POS[1] + _OFFSET, 0.5),  # +Y face
    4: (BASE_POS[0],            BASE_POS[1] - _OFFSET, 0.5),  # -Y face
}

ARM_JOINTS   = ['arm_base_joint', 'shoulder_joint', 'elbow_joint',
                'wrist_joint', 'gripper_left_joint', 'gripper_right_joint']
PANEL_JOINTS = ['left_panel_joint', 'right_panel_joint']

ROVER_CONFIGS = [
    ("rover_1", [0, -6, 1.5]),
    ("rover_2", [0,  0, 1.5]),
    ("rover_3", [0,  6, 1.5]),
]

# ── Reset scene ───────────────────────────────────────────────────────────
sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

CLEANUP_PREFIXES = ('rover', 'base_cube', 'charging_station', 'ground_plane', 'plant_')
for obj in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 1):
    if sim.getObjectParent(obj) != -1:
        continue
    alias = sim.getObjectAlias(obj, 0)
    if any(alias.startswith(p) for p in CLEANUP_PREFIXES):
        try:
            sim.removeModel(obj)
        except Exception:
            try:
                sim.removeObject(obj)
            except Exception:
                pass

# ── Ground ────────────────────────────────────────────────────────────────
ground = sim.createPrimitiveShape(sim.primitiveshape_plane, [500, 500, 0], 0)
sim.setObjectPosition(ground, -1, [0, 0, 0])
sim.setObjectAlias(ground, 'ground_plane')
sim.setShapeColor(ground, None, sim.colorcomponent_ambient_diffuse, [0.3, 0.25, 0.2])
sim.setObjectInt32Param(ground, sim.shapeintparam_static, 1)
sim.setObjectInt32Param(ground, sim.shapeintparam_respondable, 1)

# ── Base cube ─────────────────────────────────────────────────────────────
base_h = sim.createPrimitiveShape(
    sim.primitiveshape_cuboid, [BASE_SIZE, BASE_SIZE, BASE_SIZE], 0)
sim.setObjectPosition(base_h, -1, [BASE_POS[0], BASE_POS[1], BASE_SIZE / 2])
sim.setObjectAlias(base_h, 'base_cube')
sim.setShapeColor(base_h, None, sim.colorcomponent_ambient_diffuse, [0.15, 0.25, 0.75])
sim.setObjectInt32Param(base_h, sim.shapeintparam_static, 1)
sim.setObjectInt32Param(base_h, sim.shapeintparam_respondable, 1)

# ── Charging station pads (flat cylinders on the ground) ──────────────────
for sid, (x, y, _) in CHARGING_STATIONS.items():
    pad = sim.createPrimitiveShape(
        sim.primitiveshape_cylinder, [1.0, 1.0, 0.08], 0)
    sim.setObjectPosition(pad, -1, [x, y, 0.04])
    sim.setObjectAlias(pad, f'charging_station_{sid}')
    sim.setShapeColor(pad, None, sim.colorcomponent_ambient_diffuse, [1.0, 0.75, 0.0])
    sim.setObjectInt32Param(pad, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(pad, sim.shapeintparam_respondable, 0)
    print(f"Stacja ładowania {sid} → ({x:.1f}, {y:.1f})")

# ── Load rover models ─────────────────────────────────────────────────────
rover_handles = {}   # name → (handle, joint_map)
for name, pos in ROVER_CONFIGS:
    handle = sim.loadModel(rover_path)
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectOrientation(handle, -1, [0, 0, math.radians(-90)])
    sim.setObjectAlias(handle, name)
    all_joints = sim.getObjectsInTree(handle, sim.object_joint_type, 0)
    jmap = {sim.getObjectAlias(j, 0): j for j in all_joints}
    rover_handles[name] = (handle, jmap)
    print(f"Załadowano: {name}")

# ── Spawn plants (before startSimulation — no threads running yet) ─────────
PLANT_ROWS  = 3
PLANT_COLS  = 4
PLANT_START = (-15.0, -15.0)
PLANT_STEP  = (10.0, 10.0)

plant_positions = []
plant_names     = []
for _r in range(PLANT_ROWS):
    for _c in range(PLANT_COLS):
        plant_positions.append([
            PLANT_START[0] + _c * PLANT_STEP[0],
            PLANT_START[1] + _r * PLANT_STEP[1],
            1.0,
        ])
        plant_names.append(f"plant_{_r}_{_c}")

if os.path.exists(plant_path):
    _base = sim.loadModel(plant_path)
    _handles = [_base]
    for _ in range(len(plant_positions) - 1):
        _handles.append(sim.copyPasteObjects([_base], 1)[0])
    for _h, _pos, _name in zip(_handles, plant_positions, plant_names):
        sim.setObjectPosition(_h, -1, _pos)
        sim.setObjectAlias(_h, _name)
        print(f"Zasadzono: {_name} @ ({_pos[0]:.0f}, {_pos[1]:.0f})")
else:
    for _pos, _name in zip(plant_positions, plant_names):
        _h = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [0.4, 0.4, 0.8], 0)
        sim.setObjectPosition(_h, -1, _pos)
        sim.setObjectAlias(_h, _name)
        sim.setShapeColor(_h, None, sim.colorcomponent_ambient_diffuse, [0.15, 0.55, 0.1])
        sim.setObjectInt32Param(_h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(_h, sim.shapeintparam_respondable, 0)
        print(f"Zasadzono (prymityw): {_name}")

# ── Start simulation ──────────────────────────────────────────────────────
sim.startSimulation()
time.sleep(1.0)

# Reset arm joints only — panel joints are managed by the Rover class
for name, (handle, jmap) in rover_handles.items():
    for jname in ARM_JOINTS:
        if jname in jmap:
            h = jmap[jname]
            sim.setJointMode(h, sim.jointmode_force, 0)
            sim.setJointTargetPosition(h, 0)

time.sleep(0.5)

# ── Create Rover objects ───────────────────────────────────────────────────
spawn_map = {name: pos for name, pos in ROVER_CONFIGS}
rovers = []
with Rover._class_lock:
    for name, (handle, _) in rover_handles.items():
        r = Rover(
            sim=sim,
            handle=handle,
            spawn_coords=spawn_map[name],
            charging_stations=CHARGING_STATIONS,
            base_pos=BASE_POS + [0.0],
            panels_open=False,
        )
        rovers.append(r)

for rv in rovers:
    rv.all_rovers = rovers

print("Gotowe — możesz teraz uruchomić nav_gui.py")
