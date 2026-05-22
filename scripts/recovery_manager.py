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
        self.current_scan_row = None

        self.last_plant_positions = {}
        self.lane_x_positions = {}

        self.maneuver_stage = None
        self.rover_offsets = {}

        self.MAX_ROWS = 4

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

        self.current_scan_row = 0

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
            target_y = plant_y - 10.0
            
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
            
            self.scan_row_markers()
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

    def scan_row_markers(self, current_index=0):
        if current_index == 0:
            print("\n" + "=" * 70)
            print(f"[RECOVERY] READING MARKERS FOR ROW {getattr(self, 'current_scan_row', 0)}")
            print("=" * 70)
        
        # Warunek stopu: wszystkie łaziki w aktualnym rzędzie odpytane
        if current_index >= len(getattr(self.app, 'deployed_rovers', [])):
            print("\n" + "-" * 70)
            print(f" STAN MAPY AWARYJNEJ (PO RZĘDZIE {getattr(self, 'current_scan_row', 0)}):")
            print("-" * 70)
            if not getattr(self.app, 'emergency_map', {}):
                print(" [EMPTY] No coordinates saved yet.")
            else:
                for marker_id, data in self.app.emergency_map.items():
                    plant_name = data[2]
                    x_coord = data[0]
                    y_coord = data[1]
                    print(f" > Roślina: {plant_name:12} | ArUco ID: {marker_id:3} | X= {x_coord:6.2f}, Y= {y_coord:6.2f}")
            print("-" * 70)
            print("=" * 70 + "\n")

            # ODPALAMY SKOK DO KOLEJNEGO RZĘDU
            self.move_next_row()
            return

        rv = self.app.deployed_rovers[current_index]
        
        if rv.status in ("arrived", "idle"):
            print(f" -> [{rv.name}] aktywacja kamery do skanowania wizyjnego...")
            
            detected_markers = self.read_aruco_from_rover_camera(rv)
            
            if detected_markers:
                print(f"    [CAMERA SUCCESS] {rv.name} fizycznie odczytał ArUco ID: {detected_markers}")
                
                # --- IDENTYFIKACJA ROŚLINY ---
                target_plant_name = "Nieznana"
                for plant in self.app.plants:
                    dist = math.hypot(plant.pos[0] - rv.pos[0], plant.pos[1] - rv.pos[1])
                    if dist <= 6.0 and plant.pos[1] > rv.pos[1] and abs(plant.pos[0] - rv.pos[0]) <= 2.0:
                        target_plant_name = plant.name
                        break
                
                # --- KLUCZOWA PĘTLA (Tutaj wywalało błąd!) ---
                for marker_id in detected_markers:
                    if marker_id not in self.app.emergency_map:
                        # Na razie hardcodowane wartości (zmienimy je później)
                        estimated_plant_x = rv.pos[0]
                        estimated_plant_y = rv.pos[1] + 10.0
                        
                        # Zapis do mapy awaryjnej
                        self.app.emergency_map[marker_id] = (estimated_plant_x, estimated_plant_y, target_plant_name)
                        
                        # Zapis do pamięci łazika pod kątem manewru geometrycznego
                        self.last_plant_positions[rv.name] = [estimated_plant_x, estimated_plant_y]
                # ---------------------------------------------
                        
            else:
                print(f"    [CAMERA BLANK] {rv.name} patrzy, ale nie widzi markerów.")

        self.app.root.after(500, lambda: self.scan_row_markers(current_index + 1))

    def _wait_for_maneuver(self):
        """Pętla czekająca na fizyczny dojazd wszystkich łazików w danym etapie."""
        all_arrived = True
        for rv in self.app.deployed_rovers:
            # Dopóki robot jest w stanie "moving", flaga jest False
            if rv.status not in ("arrived", "idle"):
                all_arrived = False
                break
                
        if all_arrived:
            if self.maneuver_stage == 4:
                # Kiedy łaziki wyrównają kamery (krok 4), wracamy do pętli skanowania
                self.scan_row_markers()
            else:
                # Jeśli to krok 1, 2 lub 3 -> przechodzimy do kolejnego kroku
                self.maneuver_stage += 1
                self._execute_maneuver_stage()
        else:
            # Jeśli wciąż jadą, sprawdź ponownie za pół sekundy
            self.app.root.after(500, self._wait_for_maneuver)

    def move_next_row(self):
        """Rozpoczyna manewr wyprzedzania z podziałem na etapy, by uniknąć nadpisywania komend."""
        self.current_scan_row += 1

        # Upewnij się, że masz ustawione self.MAX_ROWS w __init__
        if self.current_scan_row > getattr(self, 'MAX_ROWS', 4):
            print("\n" + "*" * 70)
            print("[MISSION COMPLETE] Flota zbadala wszystkie rzędy. Mapa awaryjna jest pełna!")
            print("*" * 70 + "\n")
            return

        print("\n" + "=" * 70)
        print(f"[SYSTEM] Omijanie przeszkody i jazda do RZĘDU {self.current_scan_row}...")
        print("=" * 70)
        
        # Jeśli nie masz tego w __init__, inicjalizujemy słownik do trzymania offsetów
        if not hasattr(self, 'rover_offsets'):
            self.rover_offsets = {}

        self.maneuver_stage = 1
        self._execute_maneuver_stage()

    def _execute_maneuver_stage(self):
        """Zarządza kolejnymi krokami geometrycznymi. Pomiędzy krokami system czeka na dojazd."""
        
        if self.maneuver_stage == 1:
            print("[MANEWR 1/4] Odjazd w prawo (Wyliczanie kąta 45 stopni)")
            for rv in self.app.deployed_rovers:
                # Zapisujemy pas startowy przed ruchem
                self.lane_x_positions[rv.name] = rv.pos[0]
                plant_pos = self.last_plant_positions.get(rv.name)
                
                if plant_pos:
                    dist_to_plant = abs(plant_pos[1] - rv.pos[1])
                    offset_x = dist_to_plant
                else:
                    print(f"    [WARNING] Brak danych o rośliny dla {rv.name}! Zakładam offset 2m.")
                    offset_x = 2.0
                
                # Zapisujemy wyliczony offset dla tego łazika, by użyć go w kroku 2!
                self.rover_offsets[rv.name] = offset_x
                
                target_x = rv.pos[0] + offset_x
                print(f"    -> [{rv.name}] Odjeżdżam {offset_x:.2f}m w prawo.")
                # heading=0.0 -> wschód
                rv.go_to(target_x, rv.pos[1], heading=0.0)
                
            # Po wydaniu komend, czekamy aż łaziki tam dojadą!
            self.app.root.after(1000, self._wait_for_maneuver)

        elif self.maneuver_stage == 2:
            print("[MANEWR 2/4] Jazda w górę pola")
            for rv in self.app.deployed_rovers:
                # Odzyskujemy wyliczony wcześniej offset
                offset_x = self.rover_offsets.get(rv.name, 2.0)
                
                # Używamy Twojej matematyki: 2 * offset_x
                target_y = rv.pos[1] + (2 * offset_x)
                print(f"    -> [{rv.name}] Odjeżdżam {2*offset_x:.2f}m w górę.")
                # heading=math.pi/2 -> północ
                rv.go_to(rv.pos[0], target_y, heading=math.pi / 2)
                
            self.app.root.after(1000, self._wait_for_maneuver)

        elif self.maneuver_stage == 3:
            print("[MANEWR 3/4] Powrót w lewo na główny pas ruchu")
            for rv in self.app.deployed_rovers:
                # Wracamy bezpiecznie do zapisanego na samym początku X z pasa ruchu
                target_x = self.lane_x_positions.get(rv.name, rv.pos[0] - 2.0)
                print(f"    -> [{rv.name}] Wracam na pas {target_x:.2f} na osi X.")
                # heading=math.pi -> zachód
                rv.go_to(target_x, rv.pos[1], heading=math.pi)
                
            self.app.root.after(1000, self._wait_for_maneuver)

        elif self.maneuver_stage == 4:
            print("[MANEWR 4/4] Wyrównanie kamer na wprost przed nowym rzędem!")
            for rv in self.app.deployed_rovers:
                # Tylko obrót na północ
                rv.go_to(rv.pos[0], rv.pos[1], heading=math.pi / 2)
            
            self.app.root.after(1000, self._wait_for_maneuver)
        


    