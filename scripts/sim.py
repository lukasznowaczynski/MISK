from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os
import math

client = RemoteAPIClient()
sim = client.getObject('sim')

# --- DEFINICJA ŚCIEŻEK ---
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Ścieżka do Twojej bazowej sceny (upewnij się, że plik tam jest!)
scene_path = os.path.join(base_dir, "scenes", "powierzchnia_marsa.ttt") 
rover_path = os.path.join(base_dir, "models", "rover.ttm")
plant_path = os.path.join(base_dir, "models", "plant.ttm")
wiatka_path = os.path.join(base_dir, "models", "big_wiatka.ttm")

# --- RESTART SCENY ---
sim.stopSimulation() # Zatrzymaj jeśli działa
# Czekamy chwilę na zatrzymanie fizyki
import time
time.sleep(0.5) 

# Ładowanie sceny (to zamknie obecną i otworzy nową)
if os.path.exists(scene_path):
    sim.loadScene(scene_path)
    print(f"Załadowano świeżą scenę: {scene_path}")
else:
    print(f"BŁĄD: Nie znaleziono pliku sceny w: {scene_path}")
    # Opcjonalnie: jeśli nie ma sceny, czyścimy tylko obiekty (stara metoda)
    all_objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    for obj in all_objects:
        try:
            alias = sim.getObjectAlias(obj)
            if alias.startswith(("rover", "plant", "big_wiatka")):
                sim.removeObjects([obj])
        except: continue
# --- KONFIGURACJA ---
# Łaziki
n_rovers = 7             # Ile łazików zespawnować
distance_step = 7
rover_z_height = 1.5
rover_ori_deg = [0, 0, -90]

# Wiatki
wiatka_z_height = 5.0    # Wysokość spawnowania wiatki

# Rośliny (Siatka M x N)
plants_m = 5             # Liczba kolumn (X)
plants_n = 4             # Liczba wierszy (Y)
plant_dist_x = 20        # Odstęp w osi X
plant_dist_y = 20        # Odstęp w osi Y
plant_z_height = 1.0     # Wysokość spawnowania roślin
plant_start_x = -30      # Start pola uprawnego (X)
plant_start_y = 0        # Start pola uprawnego (Y)
# ---------------------

rover_ori_rad = [math.radians(x) for x in rover_ori_deg]

rover_positions = []
rover_names = []

for i in range(n_rovers):
    x_pos = i * distance_step - 25
    y_pos = -30
    rover_positions.append([x_pos, y_pos, rover_z_height])
    rover_names.append(f"rover_{i+1}")

# 2. Przygotowanie pozycji dla wiatek (jedna na każde rozpoczęte 5 łazików)
wiatka_positions = []
n_wiatki = math.ceil(n_rovers / 5)

for i in range(n_wiatki):
    # Środek wiatki nad 3-cim łazikiem w każdej grupie (indeks i*5 + 2)
    center_idx = i * 5 + 3
    x_pos = center_idx * distance_step - 30
    y_pos = -30
    wiatka_positions.append([x_pos, y_pos, wiatka_z_height])

# 3. Przygotowanie pozycji i nazw dla roślin
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

# --- SPAWNOWANIE WIATEK (NOWA SEKCJA) ---
if n_wiatki > 0:
    base_wiatka_handle = sim.loadModel(wiatka_path)
    wiatka_handles = [base_wiatka_handle]
    
    if n_wiatki > 1:
        for _ in range(n_wiatki - 1):
            copy = sim.copyPasteObjects([base_wiatka_handle], 1)
            wiatka_handles.append(copy[0])
            
    for i, (handle, pos) in enumerate(zip(wiatka_handles, wiatka_positions)):
        sim.setObjectPosition(handle, -1, pos)
        sim.setObjectAlias(handle, f"big_wiatka_{i+1}")

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
print(f"Zespawnowano {n_rovers} łazików pod {n_wiatki} wiatami oraz {total_plants} roślin.")