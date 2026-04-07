from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os
import math

client = RemoteAPIClient()
sim = client.getObject('sim')

# --- KONFIGURACJA ---
# Łaziki
n_rovers = 7             # Ile łazików zespawnować
distance_step = 10 
rover_z_height = 1.5
rover_ori_deg = [0, 0, -90]

# Rośliny (Siatka M x N)
plants_m = 5             # Liczba kolumn (X)
plants_n = 4             # Liczba wierszy (Y)
plant_dist_x = 20       # Odstęp w osi X
plant_dist_y = 20       # Odstęp w osi Y
plant_z_height = 1.0     # Wysokość spawnowania roślin
plant_start_x = -30      # Start pola uprawnego (X)
plant_start_y = 0        # Start pola uprawnego (Y)
# ---------------------

rover_ori_rad = [math.radians(x) for x in rover_ori_deg]

a_pwd = os.getcwd()
rover_path = a_pwd + "/models/rover.ttm"
plant_path = a_pwd + "/models/plant.ttm"



# 1. Przygotowanie pozycji i nazw dla łazików
rover_positions = []
rover_names = []

for i in range(n_rovers):
    x_pos = i * distance_step - 30
    y_pos = -30
    rover_positions.append([x_pos, y_pos, rover_z_height])
    rover_names.append(f"rover_{i+1}")

# 2. Przygotowanie pozycji i nazw dla roślin (Siatka M x N)
plant_positions = []
plant_names = []

for m in range(plants_m):
    for n in range(plants_n):
        x = plant_start_x + (m * plant_dist_x)
        y = plant_start_y + (n * plant_dist_y)
        plant_positions.append([x, y, plant_z_height])
        plant_names.append(f"plant_{m}_{n}")

# --- SPAWNOWANIE ŁAZIKÓW ---
base_rover_handle = sim.loadModel(rover_path)
rover_handles = [base_rover_handle]

if n_rovers > 1:
    for _ in range(n_rovers - 1):
        copy = sim.copyPasteObjects([base_rover_handle], 1)
        rover_handles.append(copy[0])

for handle, pos, name in zip(rover_handles, rover_positions, rover_names):
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectOrientation(handle, -1, rover_ori_rad)
    sim.setObjectAlias(handle, name)

# --- SPAWNOWANIE ROŚLIN (PLANTS) ---
base_plant_handle = sim.loadModel(plant_path)
plant_handles = [base_plant_handle]
total_plants = len(plant_positions)

if total_plants > 1:
    for _ in range(total_plants - 1):
        copy = sim.copyPasteObjects([base_plant_handle], 1)
        plant_handles.append(copy[0])

for handle, pos, name in zip(plant_handles, plant_positions, plant_names):
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectAlias(handle, name)

# Start symulacji
sim.startSimulation()

print("Symulacja uruchomiona!")
print(f"Zespawnowano {n_rovers} łazików oraz {total_plants} roślin ({plants_m}x{plants_n}).")