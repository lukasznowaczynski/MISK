import cv2
import numpy as np
import math

class RecoveryManager:
    def __init__(self, app):
        """
        Inicjalizacja z wstrzyknięciem zależności. 
        'app' to instancja NavApp, dzięki której mamy dostęp do list i GUI.
        """
        self.app = app

    # ── RECOVERY ────────────────────────────────────────────────   
    def recovery_mode(self):
        print("\n" + "=" * 60)
        print("[SYSTEM] URUCHAMIANIE PROCEDURY RECOVERY Z POZIOMU GUI")
        print("=" * 60)

        # 1. Odczyt i zabezpieczenie pozycji roślin w terminalu
        print("\n[KROK 1] Odczytywanie i zabezpieczanie pozycji roślin przed awarią:")
        print("-" * 50)
        for plant in self.app.plants:
            print(f"Roslina: {plant.name} -> Rzeczywiste Globalne X: {plant.pos[0]:.2f}, Y: {plant.pos[1]:.2f}")
        print("-" * 50)
        print(f"[SUKCES] Zabezpieczono dane {len(self.app.plants)} roślin.")

        # 2. Informacja o wejściu w tryb amnezji
        print("\n[KROK 2] Czyszczenie bazy mapowania... Symulacja amnezji floty.")
        print("[OK] Centralna mapa została wyczyszczona. Brak danych wejściowych.")

        # 3. Rozkaz fizycznej jazdy do bazy wraz z końcowym pozycjonowaniem i obrotem
        print("\n[KROK 3] Nakaz fizycznego powrotu do baz stacji dokujących dla wszystkich łazików...")
        print("-" * 50)
        
        for rv in self.app.rovers:
            print(f" -> [{rv.name}]: Zamykanie paneli i powrót na stację z obrotem końcowym.")
            
            if getattr(rv, 'panels_open', False):
                rv.close_panels()
            
            rv.go_to_base()

        print("-" * 50)
        print("[SYSTEM] Wszystkie łaziki zmierzają do wiatki.")
        print("[SYSTEM] Uruchamianie automatycznego monitora parkowania floty...")
        
        # ODPALAMY MONITOR ZJAZDU DO BAZY (odnosząc się do roota z aplikacji głównej)
        self.app.root.after(1000, self._wait_for_base_arrival)

    def _wait_for_base_arrival(self):
        # Sprawdzamy stan wszystkich łazików
        all_parked = True
        for rv in self.app.rovers:
            if rv.status not in ("arrived", "idle", "charging"):
                all_parked = False
                break
        
        if all_parked:
            print("\n" + "=" * 60)
            print("[SUKCES] Cała flota bezpiecznie zaparkowała i zsynchronizowała się w bazie!")
            print("[SYSTEM] Rozpoczynam automatyczną procedurę mapowania awaryjnego...")
            print("=" * 60)
            
            # AUTOMATYCZNY START PROFILU RECOVERY MAPPING
            self._deploy_fleet_to_start_formation()
        else:
            current_states = [f"{rv.name}:{rv.status}" for rv in self.app.rovers]
            print(f"[Monitor Parkowania]: Oczekiwanie... ({', '.join(current_states)})")
            self.app.root.after(1000, self._wait_for_base_arrival)

    def _deploy_fleet_to_start_formation(self):
        print("\n" + "=" * 60)
        print("[RECOVERY] DEPLOYING FLEET TO START FORMATION (ROW 0)")
        print("=" * 60)

        row_0_plants = []
        for plant in self.app.plants:
            try:
                parts = plant.name.split('_')
                plant_row = int(parts[2])
                plant_col = int(parts[1])
                
                if plant_row == 0:
                    row_0_plants.append((plant_col, plant))
            except (IndexError, ValueError):
                continue
        
        row_0_plants.sort(key=lambda item: item[0])
        column_count = len(row_0_plants)
        print(f"[INFO] Detected {column_count} columns of plants in row 0.")
        print(f"[INFO] Assigning exactly {column_count} rovers to the mission.")
        print("-" * 50)

        self.app.deployed_rovers = []

        for i in range(column_count):
            if i >= len(self.app.rovers):
                print("[WARNING] Not enough rovers to cover all detected plant columns!")
                break
                
            rv = self.app.rovers[i]
            _, plant = row_0_plants[i] 
            
            plant_x = plant.pos[0]
            plant_y = plant.pos[1]
            
            target_x = plant_x
            target_y = plant_y - 15.0
            
            target_heading = math.pi / 2 
            
            print(f" -> [{rv.name}] moving to lane column #{i} [{plant.name}]. Target -> X: {target_x:.1f}, Y: {target_y:.1f}")
            
            if getattr(rv, 'panels_open', False):
                rv.close_panels()
                
            rv.go_to(target_x, target_y, heading=target_heading)
            self.app.deployed_rovers.append(rv)

        if len(self.app.rovers) > column_count:
            print("-" * 50)
            for j in range(column_count, len(self.app.rovers)):
                backup_rv = self.app.rovers[j]
                print(f" -> [{backup_rv.name}] remains at the base station as backup.")
                
        print("-" * 50)
        print("[SYSTEM] Fleet is moving to the starting lineup. Initializing formation monitor...")
        print("=" * 60 + "\n")

        self.app.root.after(1000, self._wait_for_formation_arrival)

    def _wait_for_formation_arrival(self):
        all_in_position = True
        
        for rv in getattr(self.app, 'deployed_rovers', []):
            if rv.status not in ("arrived", "idle"):
                all_in_position = False
                break
                
        if all_in_position:
            print("\n" + "=" * 60)
            print("[SUCCESS] All deployed rovers have reached the start formation layout!")
            print("[SYSTEM] Initializing physical ArUco image scanning via OpenCV...")
            print("=" * 60)
            
            self.scan_row_0_markers()
        else:
            self.app.root.after(1000, self._wait_for_formation_arrival)

    def read_aruco_from_rover_camera(self, rv):
        with rv._class_lock:
            try:
                camera_handle = getattr(rv, 'camera', None)
                if camera_handle is None or camera_handle == -1:
                    print(f"[CAMERA LINK ERROR] {rv.name} nie posiada poprawnego uchwytu self.camera!")
                    return []
                
                image_bytes, resolution = rv.sim.getVisionSensorImg(camera_handle, 0)
                
                if not image_bytes or len(image_bytes) == 0:
                    return []

                width, height = resolution[0], resolution[1]
                img = np.frombuffer(image_bytes, dtype=np.uint8)
                img.shape = (height, width, 3)
                
                img = cv2.flip(img, 0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(f"debug_{rv.name}.png", img)

                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, rejected = detector.detectMarkers(img)

                detected_ids = []
                if ids is not None:
                    detected_ids = [int(marker_id[0]) for marker_id in ids]
                    
                return detected_ids

            except Exception as e:
                print(f"[CAMERA ERROR] Błąd przetwarzania obrazu dla {rv.name}: {e}")
                return []

    def scan_row_0_markers(self, current_index=0):
        if current_index == 0:
            print("\n" + "=" * 70)
            print("[RECOVERY] READING PHYSICAL ARUCO MARKERS FROM ROVER CAMERAS")
            print("=" * 70)
        
        if current_index >= len(getattr(self.app, 'deployed_rovers', [])):
            print("\n" + "-" * 70)
            print(" CURRENT EMERGENCY MAP COORDINATES (SAVED DATA):")
            print("-" * 70)
            if not getattr(self.app, 'emergency_map', {}):
                print(" [EMPTY] No coordinates saved yet.")
            else:
                for marker_id, data in self.app.emergency_map.items():
                    plant_name = data[2]
                    x_coord = data[0]
                    y_coord = data[1]
                    print(f" > Roślina: {plant_name:12} | ArUco ID: {marker_id:3} | Współrzędne: X= {x_coord:6.2f}, Y= {y_coord:6.2f}")
            print("-" * 70)
            print("=" * 70 + "\n")
            return

        rv = self.app.deployed_rovers[current_index]
        
        if rv.status in ("arrived", "idle"):
            print(f" -> [{rv.name}] aktywacja kamery do skanowania wizyjnego...")
            
            detected_markers = self.read_aruco_from_rover_camera(rv)
            
            if detected_markers:
                print(f"    [CAMERA SUCCESS] {rv.name} fizycznie odczytał ArUco ID: {detected_markers}")
                
                target_plant_name = "Nieznana"
                for plant in self.app.plants:
                    dist = math.hypot(plant.pos[0] - rv.pos[0], plant.pos[1] - rv.pos[1])
                    if dist <= 6.0 and plant.pos[1] > rv.pos[1] and abs(plant.pos[0] - rv.pos[0]) <= 2.0:
                        target_plant_name = plant.name
                        break
                
                for marker_id in detected_markers:
                    if marker_id not in self.app.emergency_map:
                        estimated_plant_x = rv.pos[0]
                        estimated_plant_y = rv.pos[1] + 5.0
                        
                        self.app.emergency_map[marker_id] = (estimated_plant_x, estimated_plant_y, target_plant_name)
                        
            else:
                print(f"    [CAMERA BLANK] {rv.name} patrzy, ale nie widzi markerów.")

        self.app.root.after(500, lambda: self.scan_row_0_markers(current_index + 1))

    