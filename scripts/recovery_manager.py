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
            
            target_x = plant_x - 2.0
            target_y = plant_y - 5.0
            
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

                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, rejected = detector.detectMarkers(img)

                detected_data = {}
                
                if ids is not None:
                    # 1. PARAMETRY WEWNĘTRZNE KAMERY (Symulacja FOV 100 stopni, 256x256 px)
                    fov_rad = rv.sim.getObjectFloatParam(camera_handle, rv.sim.visionfloatparam_perspective_angle)
                    cx = width / 2.0
                    cy = height / 2.0
                    
                    # Wyliczenie ogniskowej z dynamicznych wartości
                    focal_length = cx / math.tan(fov_rad / 2.0)

                    print(f"FOV_RAD: {fov_rad:.2f}, FOCAL_LENGTH: {focal_length:.2f}")

                    # fov_rad = math.radians(100)
                    # focal_length = (256 / 2.0) / math.tan(fov_rad / 2.0)

                    # print(f"FOV_RAD2: {fov_rad:.2f}, FOCAL_LENGTH2: {focal_length:.2f}")
                    
                    camera_matrix = np.array([
                        [focal_length, 0, 128],
                        [0, focal_length, 128],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    
                    dist_coeffs = np.zeros((4, 1))

                    # 2. ROZMIAR FIZYCZNY MARKERA (0.3 metra wg nav_gui.py)
                    marker_size = 0.3
                    obj_points = np.array([
                        [-marker_size/2,  marker_size/2, 0],
                        [ marker_size/2,  marker_size/2, 0],
                        [ marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)

                    # 3. WYLICZANIE ODLEGŁOŚCI DLA KAŻDEGO ZNALEZIONEGO KODU
                    for i in range(len(ids)):
                        marker_id = int(ids[i][0])
                        corner = corners[i]
                        
                        success, rvec, tvec = cv2.solvePnP(obj_points, corner, camera_matrix, dist_coeffs)
                        
                        if success:
                            # tvec[0][0] to przesunięcie w lewo/prawo od środka obiektywu
                            offset_x = tvec[0][0]
                            
                            # tvec[2][0] to odległość w linii prostej przed obiektywem.
                            # Dodajemy 0.75m, bo obiektyw wisi przed środkiem masy łazika!
                            offset_z = tvec[2][0] + 0.75 
                            
                            # Zwracamy krotkę: (przesunięcie_boczne, odległość_na_wprost)
                            detected_data[marker_id] = (offset_x, offset_z)
                            
                return detected_data

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
                
                for marker_id, (offset_x, offset_z) in detected_markers.items():
                    matched_plant = None
                    
                    for plant in self.app.plants:
                        if getattr(plant, 'aruco_id', None) == marker_id:
                            matched_plant = plant
                            break
                    
                    if matched_plant:
                        target_plant_name = matched_plant.name
                        
                        # Wektor "w przód" (względem tego, jak obrócony jest łazik)
                        forward_x = math.cos(rv.heading)
                        forward_y = math.sin(rv.heading)
                        
                        # Wektor "w prawo" (obrót wektora w przód o 90 stopni / -pi/2)
                        right_x = math.sin(rv.heading)
                        right_y = -math.cos(rv.heading)
                        
                        # Rzutujemy lokalne odczyty z kamery na globalną mapę Coppelii
                        plant_x = rv.pos[0] + (offset_z * forward_x) + (offset_x * right_x)
                        plant_y = rv.pos[1] + (offset_z * forward_y) + (offset_x * right_y)
                        # ------------------------------------------
                        
                        print(f"    [CV2 SUCCESS] Wykryto {target_plant_name} (Z: {offset_z:.2f}m przed łazikiem, X_boczne: {offset_x:.2f}m)")
                        print(f"    -> Wyliczona idealna pozycja globalna: X: {plant_x:.2f}, Y: {plant_y:.2f}")
                    else:
                        # Fallback bezpieczeństwa, jeśli baza byłaby pusta
                        target_plant_name = f"Nieznana (ArUco {marker_id})"
                        
                        plant_x = None
                        plant_y = None
                        print(f"    [MATCH WARNING] Wykryto ArUco {marker_id}, ale brak takiej rośliny w bazie danych!")

                    # Zapis do mapy awaryjnej (jeśli jeszcze jej nie zapisano)
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (plant_x, plant_y, target_plant_name)
                    
                    # Zapisujemy IDEALNĄ pozycję do pamięci manewru geometrycznego 45 stopni!
                    self.last_plant_positions[rv.name] = [plant_x, plant_y]
                # --------------------------------------------
                        
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

    def do_photo(self):
        print("\n" + "=" * 70)
        print("[PHOTO MODE] Capturing images from all deployed rovers...")
        print("=" * 70)
        
        rovers = getattr(self.app, 'deployed_rovers', [])
        
        if not rovers:
            print("[WARNING] Brak wdrożonych łazików! Najpierw wyślij flotę, by móc zrobić zdjęcia.")
            rovers = getattr(self.app, 'rovers', [])
            

        for rv in rovers:
            print(f"\n[DEBUG] ---> Start iteracji dla: {rv.name}")
            print(f"[DEBUG] [{rv.name}] Czekam na dostęp do wątku (rv._class_lock)...")
            
            with rv._class_lock:
                print(f"[DEBUG] [{rv.name}] Uzyskano dostęp do wątku! Wchodzę w try...")
                try:
                    camera_handle = getattr(rv, 'camera', None)
                    print(f"[DEBUG] [{rv.name}] Odczytany uchwyt kamery: {camera_handle}")
                    
                    if camera_handle is None or camera_handle == -1:
                        print(f"    [CAMERA LINK ERROR] {rv.name} nie posiada poprawnego uchwytu kamery!")
                        continue
                    
                    print(f"[DEBUG] [{rv.name}] Wywołuję rv.sim.getVisionSensorImg... (Tutaj może się zawiesić)")
                    image_bytes, resolution = rv.sim.getVisionSensorImg(camera_handle, 0)
                    print(f"[DEBUG] [{rv.name}] Sukces API! Długość odebranego bufora: {len(image_bytes) if image_bytes else 'Brak'}")
                    
                    if not image_bytes or len(image_bytes) == 0:
                        print(f"    [WARNING] Pusty bufor kamery dla {rv.name}.")
                        continue

                    print(f"[DEBUG] [{rv.name}] Rozpoczynam przetwarzanie obrazu numpy/OpenCV...")
                    width, height = resolution[0], resolution[1]
                    img = np.frombuffer(image_bytes, dtype=np.uint8)
                    img.shape = (height, width, 3)
                    
                    img = cv2.flip(img, 0)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    print(f"[DEBUG] [{rv.name}] Zapisuję plik na dysku...")
                    cv2.imwrite(f"debug_{rv.name}.png", img)
                    print(f"    -> Sukces! Zapisano: debug_{rv.name}.png")
                    
                except Exception as e:
                    print(f"    [CAMERA ERROR] Błąd przetwarzania obrazu dla {rv.name}: {e}")
                    continue
            
            print(f"[DEBUG] <--- Koniec iteracji dla: {rv.name}. Zwalniam locka.\n")
        


    