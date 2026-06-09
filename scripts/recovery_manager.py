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

        self.state = "IDLE"
        self.action_in_progress = False  # Flaga blokująca dublowanie komend
        self.scout = None

        
        self.anchor_pos = None  # (X, Y) pierwszej rośliny (Kotwicy)
        self.grid_dy = None     # Odstęp między roślinami w rzędzie
        self.grid_dx = None     # Odstęp między rzędami
        self.x_min = None       # Współrzędna X pierwszego (lewego) rzędu


    # ── RECOVERY ────────────────────────────────────────────────   
    def recovery_mode(self):
        print("\n" + "=" * 60)
        print("[SYSTEM] URUCHAMIANIE PROCEDURY RECOVERY")
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
            self.state = "INIT"
            self.action_in_progress = False
            self._state_machine_loop()
        else:
            current_states = [f"{rv.name}:{rv.status}" for rv in self.app.rovers]
            print(f"[Monitor Parkowania]: Oczekiwanie... ({', '.join(current_states)})")
            self.app.root.after(1000, self._wait_for_base_arrival)


    def _state_machine_loop(self):
        if self.state == "INIT":
            self._deploy_scout()
        elif self.state == "DUAL_ADVANCE":
            self._state_dual_advance()
        elif self.state == "SPLIT_DIVERGE":
            self._state_split_diverge()
        elif self.state == "FIND_EDGES":    
            self._state_find_edges()
        elif self.state == "DEPLOY_FLEET":
            self._state_find_dy()
        elif self.state == "CORRIDOR_SCAN":
            self._state_corridor_scan()
        elif self.state == "CORR":
            print("\n[SUKCES] Faza podwójnego zwiadu zakończona!")
            return

        self.app.root.after(300, self._state_machine_loop)

    def _deploy_scout(self):
        print("\n" + "=" * 60)
        print("[SYSTEM] START MAPOWANIA AWARYJNEGO")
        print("=" * 60)

        self.app.emergency_map = {}
        
        # Wyznaczamy DWÓCH zwiadowców (zakładamy, że to rover_1 i rover_2)
        self.scout_A = self.app.rovers[0]  # Pojedzie docelowo w LEWO
        self.scout_B = self.app.rovers[1]  # Pojedzie docelowo w PRAWO
        
        self.state = "DUAL_ADVANCE"
        self.action_in_progress = False

    def read_aruco_from_rover_camera(self, rv, camera_name="front"):
        with rv._class_lock:
            try:
                if camera_name == "front":
                    camera_handle = getattr(rv, 'camera', None) # Upewnij się, że główna kamera nazywa się rv.camera
                    CAMERA_OFFSET = 0.75
                elif camera_name == "left":
                    camera_handle = getattr(rv, 'camera_left', None)
                    CAMERA_OFFSET = 0.35
                elif camera_name == "right":
                    camera_handle = getattr(rv, 'camera_right', None)
                    CAMERA_OFFSET = 0.35
                else:
                    return {}
                
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

                    
                    camera_matrix = np.array([
                        [focal_length, 0, cx],
                        [0, focal_length, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    
                    dist_coeffs = np.zeros((4, 1))

                    # 2. ROZMIAR FIZYCZNY MARKERA (0.3 metra wg nav_gui.py)
                    marker_size = 0.3 * (392.0 / 512.0)
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
                            offset_z = tvec[2][0] + CAMERA_OFFSET 
                            
                            # Zwracamy krotkę: (przesunięcie_boczne, odległość_na_wprost)
                            detected_data[marker_id] = (offset_x, offset_z)
                            
                return detected_data

            except Exception as e:
                print(f"[CAMERA ERROR] Błąd przetwarzania obrazu dla {rv.name}: {e}")
                return []
            
    def _local_to_global(self, rv, offset_x, offset_z, camera_name="front"):
        """Przelicza offsety prosto z obiektywu na globalne współrzędne X, Y na mapie."""
        
        # 1. Ustalenie absolutnego kąta patrzenia kamery
        if camera_name == "left":
            cam_heading = rv.heading + (math.pi / 2.0)  # +90 stopni
        elif camera_name == "right":
            cam_heading = rv.heading - (math.pi / 2.0)  # -90 stopni
        else: # front
            cam_heading = rv.heading

        # 2. Wektory kierunkowe
        forward_x = math.cos(cam_heading)
        forward_y = math.sin(cam_heading)
        right_x = math.sin(cam_heading)
        right_y = -math.cos(cam_heading)

        # 3. Rzutowanie na globalną mapę Coppelii
        g_x = rv.pos[0] + (offset_z * forward_x) + (offset_x * right_x)
        g_y = rv.pos[1] + (offset_z * forward_y) + (offset_x * right_y)

        return g_x, g_y

    def _state_dual_advance(self):
        if not self.action_in_progress:
            self.scout_A.go_to(self.scout_A.pos[0], self.scout_A.pos[1] + 50.0, heading=math.pi/2)
            self.scout_B.go_to(self.scout_B.pos[0], self.scout_B.pos[1] + 50.0, heading=math.pi/2)
            
            self.action_in_progress = True
            
            # 1. Inicjalizujemy zunifikowane zmienne
            self.scout_A_plant_pos = None
            self.scout_B_plant_pos = None

        # 2. Sprawdzamy, czy zmienna jest pusta
        if not self.scout_A_plant_pos:
            det_A = self.read_aruco_from_rover_camera(self.scout_A)
            if det_A:
                marker_id, (offset_x, offset_y) = next(iter(det_A.items()))
                print(f"Offset Y dla Scout A: {offset_y:.2f} m")
                if offset_y <= 15.0:
                    self.scout_A.stop()

                    g_x = self.scout_A.pos[0] + offset_x
                    g_y = self.scout_A.pos[1] + offset_y
                    
                    # Zapisujemy dane do tej samej zmiennej, którą sprawdzaliśmy wyżej
                    self.scout_A_plant_pos = (marker_id, self.scout_A.pos[0] + offset_x, self.scout_A.pos[1] + offset_y)
                    print(f"[ZWIAD A] {self.scout_A.name} wykrył roślinę, ID: {marker_id} (X={self.scout_A_plant_pos[1]:.2f}, Y={self.scout_A_plant_pos[2]:.2f})")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)
                        print(f"[MAPA] Zarejestrowano pierwszą roślinę (ID={marker_id}) w bazie danych.")

        # Analogicznie dla Scout B
        if not self.scout_B_plant_pos:
            det_B = self.read_aruco_from_rover_camera(self.scout_B)
            if det_B:
                marker_id, (offset_x, offset_y) = next(iter(det_B.items()))
                print(f"Offset Y dla Scout B: {offset_y:.2f} m")
                if offset_y <= 15.0:
                    self.scout_B.stop()

                    g_x = self.scout_B.pos[0] + offset_x
                    g_y = self.scout_B.pos[1] + offset_y
                    
                    self.scout_B_plant_pos = (marker_id, self.scout_B.pos[0] + offset_x, self.scout_B.pos[1] + offset_y)
                    print(f"[ZWIAD B] {self.scout_B.name} wykrył roślinę, ID: {marker_id} (X={self.scout_B_plant_pos[1]:.2f}, Y={self.scout_B_plant_pos[2]:.2f})")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)
                        print(f"[MAPA] Zarejestrowano pierwszą roślinę (ID={marker_id}) w bazie danych.")

        # 3. Odwołujemy się do zmiennych RecoveryManagera, bez kropki po scout_A
        if self.scout_A_plant_pos and self.scout_B_plant_pos:
            self.action_in_progress = False
            self.state = "SPLIT_DIVERGE"

    def _state_split_diverge(self):
        """STAN 2: Szukanie pierwszej sąsiedniej rośliny w celu ustalenia DX (odstępu między rzędami)."""
        if not hasattr(self, 'dx_A'):
            self.dx_A = None
            self.dx_B = None
            self.hop_dist = 6.0
            self.scout_A_hopping = False
            self.scout_B_hopping = False
            
            # ZABEZPIECZENIE: Zapisujemy pozycje startowe na wypadek, gdyby któryś nigdy nie znalazł DX
            self.scout_A_current_x = self.scout_A_plant_pos[1]
            self.scout_B_current_x = self.scout_B_plant_pos[1]
            self.search_dx_active = True

            self.scout_A_blank_seen = False
            self.scout_B_blank_seen = False
            

        # --- LOGIKA DLA ŁAZIKA A (Jedzie w LEWO / na Zachód, patrzy PRAWĄ kamerą) ---
        if self.dx_A is None and self.search_dx_active:
            if not self.scout_A_hopping:
                target_x = self.scout_A.pos[0] - self.hop_dist
                self.scout_A.go_to(target_x, self.scout_A.pos[1], heading=math.pi)
                self.scout_A_hopping = True
            
            elif self.scout_A.status != "moving":
                det_A = self.read_aruco_from_rover_camera(self.scout_A, camera_name="right")
                if det_A:
                    for marker_id, (offset_x, offset_z) in det_A.items():
                        if marker_id != self.scout_A_plant_pos[0]:
                            
                            # MAGIA TUTAJ: Tłumaczymy offsety kamery na mapę globalną!
                            g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, "right")
                            if marker_id not in self.app.emergency_map:
                                self.app.emergency_map[marker_id] = (g_x, g_y)
                                print(f"[MAPA] Dodano roślinę ID: {marker_id} na pozycji (X: {g_x:.2f}, Y: {g_y:.2f})")
                                
                            self.dx_A = abs(g_x - self.scout_A_plant_pos[1])
                            self.scout_A_current_x = g_x 
                            print(f"[DX ZNALEZIONE] Scout A zmierzył odstęp: {self.dx_A:.2f} m")
                            break
                else:
                    self.scout_A_blank_seen = True
                
                if self.dx_A is None:
                    self.scout_A_hopping = False

        # --- LOGIKA DLA ŁAZIKA B (Jedzie w PRAWO / na Wschód, patrzy LEWĄ kamerą) ---
        if self.dx_B is None and self.search_dx_active:
            if not self.scout_B_hopping:
                target_x = self.scout_B.pos[0] + self.hop_dist
                self.scout_B.go_to(target_x, self.scout_B.pos[1], heading=0.0)
                self.scout_B_hopping = True
            
            elif self.scout_B.status != "moving":
                det_B = self.read_aruco_from_rover_camera(self.scout_B, camera_name="left")
                if det_B:
                    for marker_id, (offset_x, offset_z) in det_B.items():
                        if marker_id != self.scout_B_plant_pos[0]:
                            
                            # MAGIA TUTAJ: Tłumaczymy offsety kamery na mapę globalną!
                            g_x, g_y = self._local_to_global(self.scout_B, offset_x, offset_z, "left")
                            
                            if marker_id not in self.app.emergency_map:
                                self.app.emergency_map[marker_id] = (g_x, g_y)
                                print(f"[MAPA] Dodano roślinę ID: {marker_id} na pozycji (X: {g_x:.2f}, Y: {g_y:.2f})")
                             
                            self.dx_B = abs(g_x - self.scout_B_plant_pos[1])
                            self.scout_B_current_x = g_x
                            print(f"[DX ZNALEZIONE] Scout B zmierzył odstęp: {self.dx_B:.2f} m")
                            break
                else:
                    self.scout_B_blank_seen = True
                
                if self.dx_B is None:
                    self.scout_B_hopping = False

        # --- KONSENSUS LUB RATUNEK (INTELIGENTNE WNIOSKOWANIE O KRAWĘDZIACH) ---
        if self.dx_A is not None or self.dx_B is not None:
            self.search_dx_active = False 
            
            # Domyślnie zakładamy, że oba muszą w kolejnym etapie szukać krawędzi...
            self.scout_A_searching = True
            self.scout_B_searching = True
            
            if self.dx_A is not None and self.dx_B is not None:
                self.grid_dx = (self.dx_A + self.dx_B) / 2.0
                print(f"\n[SUKCES] OBA łaziki znalazły DX. Średnia = {self.grid_dx:.2f} m")
                
            elif self.dx_A is not None:
                self.grid_dx = self.dx_A
                self.scout_B.stop() 
                
                # ...chyba że wdrożymy nasze inteligentne wnioskowanie dla B!
                if self.scout_B_blank_seen:
                    self.x_max = self.scout_B_plant_pos[1]
                    self.scout_B_searching = False 
                
            elif self.dx_B is not None:
                self.grid_dx = self.dx_B
                self.scout_A.stop() 
                
                # ...chyba że wdrożymy nasze inteligentne wnioskowanie dla A!
                if self.scout_A_blank_seen:
                    self.x_min = self.scout_A_plant_pos[1]
                    self.scout_A_searching = False 

            print("=" * 60)
            
            self.action_in_progress = False
            
            # Czyszczenie pamięci tymczasowej przed przejściem dalej
            del self.dx_A
            del self.dx_B
            del self.search_dx_active
            
            # Odpalenie stanu rozjazdu do brzegów pola!
            self.state = "FIND_EDGES"
            

    def _state_find_edges(self):
        """STAN 3: Szukanie skrajnych krawędzi pola. Łaziki skaczą w boki o pełne grid_dx."""
        
        # Jeśli OBA łaziki znalazły swoje krawędzie - KONIEC ZWIADU
        if not getattr(self, 'scout_A_searching', True) and not getattr(self, 'scout_B_searching', True):
            print("\n" + "=" * 60)
            print("[SUKCES FULL] Wymiary pola poziome zmapowane!")
            print(f" -> Skrajnie lewy rząd:  X = {self.x_min:.2f} m")
            print(f" -> Skrajnie prawy rząd: X = {self.x_max:.2f} m")
            print(f" -> Szerokość alejek:    DX = {self.grid_dx:.2f} m")
            print("=" * 60)
            
            self.action_in_progress = False
            self.state = "DEPLOY_FLEET" 
            return

        # Jeśli łaziki nie są w trakcie ruchu, planujemy kolejny skok
        if not self.action_in_progress:
            # Skok łazika A (w lewo o pełne grid_dx)
            if self.scout_A_searching:
                target_A_x = self.scout_A_current_x - self.grid_dx
                # print(f"[EDGES] Scout A skok LEWO na X: {target_A_x:.2f} m")
                self.scout_A.go_to(target_A_x, self.scout_A.pos[1], heading=math.pi)
                
            # Skok łazika B (w prawo o pełne grid_dx)
            if self.scout_B_searching:
                target_B_x = self.scout_B_current_x + self.grid_dx
                # print(f"[EDGES] Scout B skok PRAWO na X: {target_B_x:.2f} m")
                self.scout_B.go_to(target_B_x, self.scout_B.pos[1], heading=0.0)
                
            self.action_in_progress = True

        # Sprawdzamy status ruchu (czy łaziki już dojechały na miejsce)
        wait_for_A = self.scout_A_searching and self.scout_A.status == "moving"
        wait_for_B = self.scout_B_searching and self.scout_B.status == "moving"

        # Kiedy łaziki wcisną hamulec po skoku:
        if not wait_for_A and not wait_for_B:
            
            # --- Sprawdzamy Scout A (Kamera Prawa) ---
            if self.scout_A_searching:
                det_A = self.read_aruco_from_rover_camera(self.scout_A, "right")
                if det_A:
                    # Znaleziono roślinę - aktualizujemy krawędź i przygotowujemy się do kolejnego skoku
                    marker_id, (offset_x, offset_z) = next(iter(det_A.items()))


                    g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, "right")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)
                        print(f"[MAPA] Dodano roślinę ID: {marker_id} na pozycji (X: {g_x:.2f}, Y: {g_y:.2f})")
                    self.scout_A_current_x = g_x
                    # print(f" -> Scout A (ID: {marker_id}). Skaczę dalej w lewo.")
                    self.action_in_progress = False 
                else:
                    self.x_min = self.scout_A_current_x
                    self.scout_A_searching = False
                    
                    # POWRÓT NA KRAWĘDŹ:
                    # print(f" -> Pusto! Wracam na X_min: {self.x_min:.2f}")
                    self.scout_A.go_to(self.x_min, self.scout_A.pos[1], heading=math.pi)
                    
                    self.scout_A.stop()
                    # print(f"\n[*] Scout A ustawił się na LEWEJ krawędzi pola.")
                    self.action_in_progress = False

            # --- Sprawdzamy Scout B (Kamera Lewa) ---
            if self.scout_B_searching:
                det_B = self.read_aruco_from_rover_camera(self.scout_B, "left")
                if det_B:
                    # Znaleziono roślinę - aktualizujemy krawędź i przygotowujemy się do kolejnego skoku
                    marker_id, (offset_x, offset_z) = next(iter(det_B.items()))
                    g_x, g_y = self._local_to_global(self.scout_B, offset_x, offset_z, "left")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)
                        print(f"[MAPA] Dodano roślinę ID: {marker_id} na pozycji (X: {g_x:.2f}, Y: {g_y:.2f})")
                    self.scout_B_current_x = g_x
                    # print(f" -> Scout B (ID: {marker_id}). Skaczę dalej w prawo.")
                    self.action_in_progress = False 
                else:
                    # Pusto! Prawa krawędź ustrzelona. Cofamy łazika do ostatniej dobrej pozycji!
                    self.x_max = self.scout_B_current_x
                    self.scout_B_searching = False
                    
                    # POWRÓT NA KRAWĘDŹ:
                    # print(f" -> Pusto! Wracam na X_max: {self.x_max:.2f}")
                    self.scout_B.go_to(self.x_max, self.scout_B.pos[1], heading=0.0)
                    
                    self.scout_B.stop()
                    # print(f"\n[*] Scout B ustawił się na PRAWEJ krawędzi pola.")
                    self.action_in_progress = False


    def _state_find_dy(self):
        """STAN: Empiryczne sondowanie pionowego rozstawu rzędów (DY) metodą skokową."""
        
        # --- 1. INICJALIZACJA ZMIENNYCH DLA SONDOWANIA ---
        if not hasattr(self, 'dy_hopping'):
            print("\n[PROBE_DY] Rozpoczynam procedurę sondowania pionowego...")
            self.dy_hopping = False
            self.dy_hop_dist = 2.0  # Długość jednego skoku sondy w głąb pola
            self.dy_probing_active = True
            self.action_in_progress = True
            
            # NOWOŚĆ: Flaga sprawdzająca, czy łazik ustawił się już w nowym korytarzu
            self.dy_aligned = False 

        # --- 1B. ETAP DOPASOWANIA: ODJAZD O 5M W LEWO OD ROŚLINY ID 10 ---
        if self.dy_probing_active and not self.dy_aligned:
            if not hasattr(self, 'dy_aligning_started'):
                # Sprawdzamy, czy roślina o ID 10 została już zapisana w naszej mapie offline
                if 10 in self.app.emergency_map:
                    plant_10_x, plant_10_y = self.app.emergency_map[10]
                    
                    # Obliczamy pozycję korytarza: 5 metrów na lewo od linii rośliny ID 10
                    target_x = plant_10_x - 5.0 
                    
                    # print(f"[ALIGN] Zjeżdżam do korytarza obok. Cel X: {target_x:.2f} m (5m w lewo od ID 10)")
                    
                    # Wysyłamy łazika na nowy X. Zostawiamy obecny Y. 
                    # Od razu ustawiamy heading=math.pi/2, żeby po dojechaniu stał przodem do kierunku jazdy (Północ).
                    self.scout_A.go_to(target_x, self.scout_A.pos[1], heading=self.scout_A.heading)
                    self.dy_aligning_started = True
                else:
                    # print("[WARN] Brak rośliny o ID 10 w pamięci mapy! Pomijam wyrównanie boczne.")
                    self.dy_aligned = True
            
            # Czekamy, aż łazik zakończy fizyczny dojazd w bok do nowego korytarza
            elif self.scout_A.status != "moving":
                # print("[ALIGN] Łazik zaparkował w lewym korytarzu. Rozpoczynam skoki w głąb pola.")
                self.dy_aligned = True
                del self.dy_aligning_started
            
            # Przerywamy bieżący obieg pętli i czekamy na kolejny (aż skończy jechać w bok)
            return 

        # --- 2. LOGIKA SKOKOWA DLA ŁAZIKA A (Uruchamia się DOPIERO po ustawieniu w korytarzu) ---
        if self.dy_probing_active and self.dy_aligned:
            
            # Krok A: Jeśli nie wykonuje skoku, zlecamy ruch do przodu
            if not self.dy_hopping:
                target_y = self.scout_A.pos[1] + self.dy_hop_dist
                # print(f"[PROBE_DY] Skok w głąb pola na Y: {target_y:.2f} m")
                
                # heading=math.pi/2 to obrót na "Północ" (wzdłuż osi Y)
                self.scout_A.go_to(self.scout_A.pos[0], target_y, heading=math.pi/2)
                self.dy_hopping = True
                
            # Krok B: Jeśli łazik zahamował po skoku, odczytujemy kamerę
            elif self.scout_A.status != "moving":
                det = self.read_aruco_from_rover_camera(self.scout_A, camera_name="front")
                
                if det:
                    for marker_id, (offset_x, offset_z) in det.items():
                        # Szukamy INNEGO id niż to, przy którym wystartowaliśmy
                        if marker_id != self.scout_A_plant_pos[0]:
                            
                            print(f"\n[DETECTION] {self.scout_A.name} wykrył NOWY marker (ID: {marker_id}).")
                            
                            # Przeliczamy na globalne Y
                            g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, "front")
                            
                            # DY to różnica między Y nowej rośliny, a Y pierwszej rośliny (zapisanej w [2])
                            self.grid_dy = abs(g_y - self.scout_A_plant_pos[2])
                            
                            print(f"[SUKCES] Wykryto pionowy rozstaw rzędów: DY = {self.grid_dy:.2f} m")

                            self.new_row_start_x = g_x
                            self.new_row_start_y = g_y
                            
                            self.dy_probing_active = False
                            
                            break
                            
                    # Jeśli widzi roślinę, ale to stara roślina, musi skakać dalej
                    if self.dy_probing_active:
                        self.dy_hopping = False
                        
                else:
                    # Kamera zwróciła pustkę - skaczemy dalej!
                    # print(f"    [CAMERA BLANK] {self.scout_A.name} nie widzi nowego rzędu. Kontynuuję jazdę.")
                    self.dy_hopping = False

        # --- 3. ZAKOŃCZENIE STANU I SPRZĄTANIE ---
        if not self.dy_probing_active:
            print("=" * 60)
            self.scout_A.stop()
            self.action_in_progress = False
            
            # Sprzątamy zmienne tymczasowe
            del self.dy_hopping
            del self.dy_hop_dist
            del self.dy_probing_active
            del self.dy_aligned # Sprzątamy nową flagę
            
            # Ważne: Przechodzimy do DONE, żeby nie wpaść w pętlę!
            self.state = "CORRIDOR_SCAN"

            print("\n" + "=" * 50)
            print("AKTUALNA MAPA AWARYJNA (EMERGENCY MAP)")
            print("=" * 50)
            
            if not self.app.emergency_map:
                print("  [!] Mapa jest obecnie pusta. Brak danych.")
            else:
                # Używamy sorted(), żeby rośliny wyświetlały się po kolei według ID
                for marker_id, (x, y) in sorted(self.app.emergency_map.items()):
                    print(f"ID: {marker_id:2d}  |  X: {x:6.2f} m  |  Y: {y:6.2f} m")
                    
            print("=" * 50 + "\n")

    
    def _state_corridor_scan(self):
        """STAN: Prostopadłe (L-kształtne) wyrównanie do korytarza i skokowe skanowanie rzędów w prawo (co DX)."""
        
        # --- 1. INICJALIZACJA ---
        if not hasattr(self, 'scan_phase'):
            print("\n" + "=" * 60)
            print("[CORRIDOR SCAN] Rozpoczynam wjazd do korytarza!")
            print("=" * 60)
            
            # 1. Pozycja Y: Dokładnie w połowie między starym a nowym rzędem
            self.corridor_y = self.new_row_start_y - (self.grid_dy / 2.0)
            
            # 2. Pozycja X: Na wprost pierwszej rośliny z nowego rzędu
            self.current_scan_x = self.new_row_start_x
            
            # Zmieniamy fazę startową na jazdę tylko w osi Y
            self.scan_phase = "ALIGN_Y" 
            self.scan_hopping = False
            self.action_in_progress = True

        # --- 2A. FAZA ALIGN_Y: Najpierw jedziemy "w górę" do linii korytarza ---
        if self.scan_phase == "ALIGN_Y":
            if not self.scan_hopping:
                # Zmieniamy TYLKO pozycję Y. X zostaje bez zmian. 
                # heading=math.pi/2 upewnia się, że łazik patrzy prosto w kierunku jazdy (na Północ)
                self.scout_A.go_to(self.scout_A.pos[0], self.corridor_y, heading=math.pi/2)
                self.scan_hopping = True
                
            elif self.scout_A.status != "moving":
                self.scan_phase = "ALIGN_X"
                self.scan_hopping = False

        # --- 2B. FAZA ALIGN_X: Następnie jedziemy "w prawo" na wprost rośliny ---
        elif self.scan_phase == "ALIGN_X":
            if not self.scan_hopping:
                # Zmieniamy TYLKO pozycję X. Y zostaje z korytarza.
                # heading=0.0 odwraca łazik na Wschód (w prawo)
                self.scout_A.go_to(self.current_scan_x, self.corridor_y, heading=0.0)
                self.scan_hopping = True
                
            elif self.scout_A.status != "moving":
                self.scan_phase = "SCAN_STEP"
                self.scan_hopping = False

        # --- 3. FAZA SCAN_STEP: Skoki w prawo o wartość DX ---
        elif self.scan_phase == "SCAN_STEP":
            if not self.scan_hopping:
                
                # A) SKANOWANIE NA POSTOJU (Kamera Lewa i Prawa)
                for cam in ["left", "right"]:
                    det = self.read_aruco_from_rover_camera(self.scout_A, camera_name=cam)
                    if det:
                        for marker_id, (offset_x, offset_z) in det.items():
                            if marker_id not in self.app.emergency_map:
                                g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, cam)
                                self.app.emergency_map[marker_id] = (g_x, g_y)
                                print(f"[MAPA] Odkryto roślinę ID: {marker_id:2d} (X: {g_x:.2f}, Y: {g_y:.2f})")
                
                # B) DECYZJA O KOLEJNYM SKOKU
                # Zostawiamy mały margines (0.1), żeby nie skakał poza krawędź x_max
                if self.current_scan_x >= self.x_max - 0.1:
                    print("\n[SCAN] Osiągnięto prawą krawędź pola (X_max)! Korytarz zmapowany.")
                    self.scan_phase = "DONE"
                else:
                    # Dodajemy DX i jedziemy wzdłuż korytarza
                    self.current_scan_x += self.grid_dx
                    self.scout_A.go_to(self.current_scan_x, self.corridor_y, heading=0.0)
                    self.scan_hopping = True
                    
            elif self.scout_A.status != "moving":
                # Gdy dojedzie na nową pozycję, zwalniamy flagę - w kolejnym takcie wykona skan!
                self.scan_hopping = False

        # --- 4. ZAKOŃCZENIE STANU ---
        elif self.scan_phase == "DONE":
            self.scout_A.stop()
            self.action_in_progress = False
            
            # Sprzątamy
            del self.scan_phase
            del self.scan_hopping
            del self.corridor_y
            del self.current_scan_x
            del self.new_row_start_x
            del self.new_row_start_y
            
            self.state = "DONE"