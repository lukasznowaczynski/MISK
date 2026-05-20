

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os
import math
import time

client = RemoteAPIClient()
sim = client.getObject('sim')

# --- DEFINICJA ŚCIEŻEK ---
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Ścieżki do modeli i sceny
scene_path = os.path.join(base_dir, "scenes", "powierzchnia_marsa.ttt") 
rover_path = os.path.join(base_dir, "models", "rover2.ttm")
plant_path = os.path.join(base_dir, "models", "plant.ttm")
wiatka_path = os.path.join(base_dir, "models", "big_wiatka.ttm")

# --- RESTART SCENY ---
sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

if os.path.exists(scene_path):
    sim.loadScene(scene_path)
    print(f"Załadowano świeżą scenę: {scene_path}")
else:
    print(f"BŁĄD: Nie znaleziono pliku sceny!")

# Usuń obiekty które mogły być zapisane w pliku sceny
SPAWN_PREFIXES = ('rover', 'plant_', 'big_wiatka')
all_objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 1)
to_remove = []
for obj in all_objects:
    if sim.getObjectParent(obj) != -1:
        continue  # tylko root-level
    alias = sim.getObjectAlias(obj, 0)
    if any(alias.startswith(p) for p in SPAWN_PREFIXES):
        to_remove.append((obj, alias))
for obj, alias in to_remove:
    sim.removeModel(obj)
    print(f"Usunięto istniejący obiekt: {alias}")

# Duży płaski teren 500x500m
for obj in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 1):
    if sim.getObjectParent(obj) == -1 and sim.getObjectAlias(obj, 0) == 'ground_plane':
        sim.removeObject(obj)

ground = sim.createPrimitiveShape(sim.primitiveshape_plane, [500, 500, 0], 0)
sim.setObjectPosition(ground, -1, [0, 0, 0])
sim.setObjectAlias(ground, 'ground_plane')
sim.setShapeColor(ground, None, sim.colorcomponent_ambient_diffuse, [0.3, 0.25, 0.2])
sim.setObjectInt32Param(ground, sim.shapeintparam_static, 1)
sim.setObjectInt32Param(ground, sim.shapeintparam_respondable, 1)

# --- KONFIGURACJA ---
n_rovers = 10
distance_step = 7
rover_z_height = 1.5
rover_ori_deg = [0, 0, -90]

wiatka_z_height = 5.0

plants_m = 5 # kolumny (X)
plants_n = 4 # wierszy (Y)
plant_dist_x = 20
plant_dist_y = 20
plant_z_height = 1.0
plant_start_x = -30
plant_start_y = 0

# --- OBLICZENIA POZYCJI ---
rover_ori_rad = [math.radians(x) for x in rover_ori_deg]
rover_positions = [[i * distance_step - 25, -30, rover_z_height] for i in range(n_rovers)]
rover_names = [f"rover_{i+1}" for i in range(n_rovers)]

n_wiatki = math.ceil(n_rovers / 5)
wiatka_positions = [[(i * 5 + 3) * distance_step - 30, -30, wiatka_z_height] for i in range(n_wiatki)]

plant_positions = []
plant_names = []
for m in range(plants_m):
    for n in range(plants_n):
        x = plant_start_x + (m * plant_dist_x)
        y = plant_start_y + (n * plant_dist_y)
        plant_positions.append([x, y, plant_z_height])
        plant_names.append(f"plant_{m}_{n}")

# --- SPAWNOWANIE ŁAZIKÓW ---
if os.path.exists(rover_path):
    base_rover_handle = sim.loadModel(rover_path)
    rover_handles = [base_rover_handle]
    for _ in range(n_rovers - 1):
        copy = sim.copyPasteObjects([base_rover_handle], 1)
        rover_handles.append(copy[0])

    for handle, pos, name in zip(rover_handles, rover_positions, rover_names):
        sim.setObjectPosition(handle, -1, pos)
        sim.setObjectOrientation(handle, -1, rover_ori_rad)
        sim.setObjectAlias(handle, name)

# --- SPAWNOWANIE WIATEK ---
if n_wiatki > 0 and os.path.exists(wiatka_path):
    base_wiatka_handle = sim.loadModel(wiatka_path)
    wiatka_handles = [base_wiatka_handle]
    for _ in range(n_wiatki - 1):
        copy = sim.copyPasteObjects([base_wiatka_handle], 1)
        wiatka_handles.append(copy[0])
            
    for i, (handle, pos) in enumerate(zip(wiatka_handles, wiatka_positions)):
        sim.setObjectPosition(handle, -1, pos)
        sim.setObjectAlias(handle, f"big_wiatka_{i+1}")

# --- SPAWNOWANIE ROŚLIN ---
if os.path.exists(plant_path):
    base_plant_handle = sim.loadModel(plant_path)
    plant_handles = [base_plant_handle]
    total_plants = len(plant_positions)

    for _ in range(total_plants - 1):
        copy = sim.copyPasteObjects([base_plant_handle], 1)
        plant_handles.append(copy[0])

    for i, (handle, pos, name) in enumerate(zip(plant_handles, plant_positions, plant_names)):
        sim.setObjectPosition(handle, -1, pos)
        sim.setObjectAlias(handle, name)
        # Python nadaje unikalną nazwę (np. plant_3_1).
        # W momencie kliknięcia PLAY, wewnętrzny skrypt Lua w nowym modelu plant.ttm
        # sam odczyta tę nazwę, zmieni ID w suwakach na poprawne i przerysuje marker na duży!
        print(f"Zespawnowano roślinę: {name}")

# Start symulacji
sim.startSimulation()
time.sleep(0.5)  # poczekaj aż skrypty Lua się zainicjalizują

# --- STEROWANIE ŁAZIKÓW ---

class RoverController:
    def __init__(self, sim, name):
        self.sim = sim
        self.name = name

    def set_velocity(self, left_vel, right_vel):
        self.sim.setFloatSignal(self.name + '_lv', left_vel)
        self.sim.setFloatSignal(self.name + '_rv', right_vel)

    def forward(self, speed=2.0):
        self.set_velocity(speed, speed)

    def backward(self, speed=2.0):
        self.set_velocity(-speed, -speed)

    def turn_left(self, speed=1.5):
        self.set_velocity(-speed, speed)

    def turn_right(self, speed=1.5):
        self.set_velocity(speed, -speed)

    def stop(self):
        self.set_velocity(0, 0)

controllers = [RoverController(sim, name) for name in rover_names]

# Przykład: jedź do przodu 3s, obróć się, jedź dalej
for ctrl in controllers:
    ctrl.forward(speed=2.0)

time.sleep(3)

for ctrl in controllers:
    ctrl.turn_right(speed=1.5)

time.sleep(1.5)

for ctrl in controllers:
    ctrl.forward(speed=2.0)

time.sleep(3)

for ctrl in controllers:
    ctrl.stop()