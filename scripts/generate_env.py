from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os
import math
import time
import cv2
import numpy as np

def setup_simulation_environment():
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    scene_path = os.path.join(base_dir, "scenes", "powierzchnia_marsa.ttt") 
    rover_path = os.path.join(base_dir, "models", "rover.ttm")
    plant_path = os.path.join(base_dir, "models", "plant1.ttm")
    wiatka_path = os.path.join(base_dir, "models", "big_wiatka.ttm")

    sim.stopSimulation()
    time.sleep(0.5) 

    if os.path.exists(scene_path):
        sim.loadScene(scene_path)
        print(f"Stworzono scene {scene_path}")
    else:
        print("BŁĄD: Nie znaleziono pliku sceny!")

    # parametry
    n_rovers = 7
    distance_step = 7
    rover_z_height = 1.5
    rover_ori_deg = [0, 0, -90]

    wiatka_z_height = 5.0
    wiatka_x_center = 10.0  
    safe_margin = 20        

    plants_m = 5  # kolumny (X)
    plants_n = 4  # wiersze (Y)
    plant_dist_x = 20
    plant_dist_y = 20
    plant_z_height = 1.0
    plant_start_x = -30
    plant_start_y = 0

    # pozycja lazikow i roslin
    rover_ori_rad = [math.radians(x) for x in rover_ori_deg]
    rover_positions = []
    half_rovers = n_rovers // 2

    for i in range(n_rovers):
        if i < half_rovers:
            x_pos = wiatka_x_center - (half_rovers - i - 1) * distance_step - safe_margin
        else:
            x_pos = wiatka_x_center + (i - half_rovers) * distance_step + safe_margin
        rover_positions.append([x_pos, -30, rover_z_height])

    rover_names = [f"rover_{i+1}" for i in range(n_rovers)]

    n_wiatki = 1
    wiatka_positions = [[wiatka_x_center, -30, wiatka_z_height]] 

    # macierz roslin
    plant_positions = []
    plant_names = []
    for m in range(plants_m):
        for n in range(plants_n):
            x = plant_start_x + (m * plant_dist_x)
            y = plant_start_y + (n * plant_dist_y)
            plant_positions.append([x, y, plant_z_height])
            plant_names.append(f"plant_{m}_{n}")

    # laziki
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

    # wiatki
    if n_wiatki > 0 and os.path.exists(wiatka_path):
        base_wiatka_handle = sim.loadModel(wiatka_path)
        wiatka_handles = [base_wiatka_handle]
        for _ in range(n_wiatki - 1):
            copy = sim.copyPasteObjects([base_wiatka_handle], 1)
            wiatka_handles.append(copy[0])
                
        for i, (handle, pos) in enumerate(zip(wiatka_handles, wiatka_positions)):
            sim.setObjectPosition(handle, -1, pos)
            sim.setObjectAlias(handle, f"big_wiatka_{i+1}")

    # spawnowanie roslin z aruco
    plant_handles = []

    if os.path.exists(plant_path):
        base_plant_handle = sim.loadModel(plant_path)
        plant_handles.append(base_plant_handle)
        total_plants = len(plant_positions)

        print(f"generowanie {total_plants} roslin")
        for _ in range(total_plants - 1):
            copy = sim.copyPasteObjects([base_plant_handle], 1)
            plant_handles.append(copy[0])
            
        # slownik aruco 5x5
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

        for i, (handle, pos, name) in enumerate(zip(plant_handles, plant_positions, plant_names)):
            sim.setObjectPosition(handle, -1, pos)
            sim.setObjectAlias(handle, name)
            
            try:
                plane_handle = sim.getObject('./aruco_plane', {'proxy': handle})
                tex_res = 512
                visual_id = i + 10
                marker_size_px = 392
                
                try:
                    base_marker = cv2.aruco.generateImageMarker(aruco_dict, visual_id, marker_size_px)
                except AttributeError:
                    base_marker = cv2.aruco.drawMarker(aruco_dict, visual_id, marker_size_px)
                marker_img = np.ones((tex_res, tex_res), dtype=np.uint8) * 255
                
                offset = (tex_res - marker_size_px) // 2
                marker_img[offset:offset+marker_size_px, offset:offset+marker_size_px] = base_marker
                marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
                marker_rgb = cv2.flip(marker_rgb, 0)
                marker_bytes = marker_rgb.tobytes()
                
                temp_shape, new_texture_id, _ = sim.createTexture("", 0, [0.15, 0.15], [1, 1], [0,0,0], 1, [tex_res, tex_res])
                sim.setShapeTexture(plane_handle, new_texture_id, 0, 0, [0.3, 0.3])
                
                sim.removeObject(temp_shape)
                sim.writeTexture(new_texture_id, 0, marker_bytes, 0, 0, tex_res, tex_res)
                sim.setObjectSizeValues(plane_handle, [0.3, 0.3, 0.001])
                
                print(f"Stworzono {name} z aruco o id:{visual_id}")
                
            except Exception as e:
                print(f"Błąd {name}: {e}")
    else:
        print(f"nie ma modeli {plant_path}")

    # Start symulacji
    sim.startSimulation()
    return client, sim