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
            self._deploy_scout()
        else:
            current_states = [f"{rv.name}:{rv.status}" for rv in self.app.rovers]
            print(f"[Monitor Parkowania]: Oczekiwanie... ({', '.join(current_states)})")
            self.app.root.after(1000, self._wait_for_base_arrival)

    def _deploy_scout(self):
        print("\n" + "=" * 60)
        print("[SYSTEM] START MAPOWANIA: PODWÓJNY ZWIAD (STEREO WIZJA)")
        print("=" * 60)

        self.app.emergency_map = {}
        
        # Wyznaczamy DWÓCH zwiadowców (zakładamy, że to rover_1 i rover_2)
        self.scout_A = self.app.rovers[0]  # Pojedzie docelowo w LEWO
        self.scout_B = self.app.rovers[1]  # Pojedzie docelowo w PRAWO
        
        print(f"[ZWIAD] Rozkaz przydzielony: {self.scout_A.name} oraz {self.scout_B.name}")
        
        self.state = "DUAL_ADVANCE"
        self.action_in_progress = False
        self._state_machine_loop()

    def _state_machine_loop(self):
        if self.state == "DUAL_ADVANCE":
            self._state_dual_advance()
        elif self.state == "SPLIT_DIVERGE":
            self._state_split_diverge()
        elif self.state == "DONE":
            print("\n[SUKCES] Faza podwójnego zwiadu zakończona!")
            return

        self.app.root.after(300, self._state_machine_loop)

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
                            offset_z = tvec[2][0] + 0.75 
                            
                            # Zwracamy krotkę: (przesunięcie_boczne, odległość_na_wprost)
                            detected_data[marker_id] = (offset_x, offset_z)
                            
                return detected_data

            except Exception as e:
                print(f"[CAMERA ERROR] Błąd przetwarzania obrazu dla {rv.name}: {e}")
                return []

    def _state_dual_advance(self):
        if not self.action_in_progress:
            print(f"[DUAL ZWIAD] Łaziki ruszają w pole. Każdy szuka swojego celu...")
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
                self.scout_A.stop()
                
                # Zapisujemy dane do tej samej zmiennej, którą sprawdzaliśmy wyżej
                self.scout_A_plant_pos = (marker_id, self.scout_A.pos[0] + offset_x, self.scout_A.pos[1] + offset_y)
                print(f"[ZWIAD A] {self.scout_A.name} wykrył roślinę, ID: {marker_id} (X={self.scout_A_plant_pos[1]:.2f}, Y={self.scout_A_plant_pos[2]:.2f})")

        # Analogicznie dla Scout B
        if not self.scout_B_plant_pos:
            det_B = self.read_aruco_from_rover_camera(self.scout_B)
            if det_B:
                marker_id, (offset_x, offset_y) = next(iter(det_B.items()))
                self.scout_B.stop()
                
                self.scout_B_plant_pos = (marker_id, self.scout_B.pos[0] + offset_x, self.scout_B.pos[1] + offset_y)
                print(f"[ZWIAD B] {self.scout_B.name} wykrył roślinę, ID: {marker_id} (X={self.scout_B_plant_pos[1]:.2f}, Y={self.scout_B_plant_pos[2]:.2f})")

        # 3. Odwołujemy się do zmiennych RecoveryManagera, bez kropki po scout_A
        if self.scout_A_plant_pos and self.scout_B_plant_pos:
            print("\n[STATUS] Oba łaziki na pozycjach. Przechodzimy do ustalenia DX.")
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
            
            print("\n[DUAL ZWIAD] Łaziki rozpoczynają żabie skoki na boki, aby ustalić DX...")

        # --- LOGIKA DLA ŁAZIKA A (Skacze w LEWO) ---
        if self.dx_A is None and self.search_dx_active:
            if not self.scout_A_hopping:
                target_x = self.scout_A.pos[0] - self.hop_dist
                self.scout_A.go_to(target_x, self.scout_A.pos[1], heading=math.pi/2)
                self.scout_A_hopping = True
            
            elif self.scout_A.status != "moving":
                det_A = self.read_aruco_from_rover_camera(self.scout_A)
                if det_A:
                    for marker_id, (g_x, g_y) in det_A.items():
                        if marker_id != self.scout_A_plant_pos[0]:
                            curr_plant_pos = (self.scout_A.pos[0] + g_x, self.scout_A.pos[1] + g_y)
                            self.dx_A = abs(curr_plant_pos[0] - self.scout_A_plant_pos[1])
                            self.scout_A_current_x = curr_plant_pos[0] 
                            print(f"[DX ZNALEZIONE] Scout A ({self.scout_A.name}) zmierzył odstęp: {self.dx_A:.2f} m")
                            break
                
                if self.dx_A is None:
                    self.scout_A_hopping = False

        # --- LOGIKA DLA ŁAZIKA B (Skacze w PRAWO) ---
        if self.dx_B is None and self.search_dx_active:
            if not self.scout_B_hopping:
                target_x = self.scout_B.pos[0] + self.hop_dist
                self.scout_B.go_to_square(target_x, self.scout_B.pos[1], heading=math.pi/2)
                self.scout_B_hopping = True
            
            elif self.scout_B.status != "moving":
                det_B = self.read_aruco_from_rover_camera(self.scout_B)
                if det_B:
                    for marker_id, (g_x, g_y) in det_B.items():
                        if marker_id != self.scout_B_plant_pos[0]:
                            curr_plant_pos = (self.scout_B.pos[0] + g_x, self.scout_B.pos[1] + g_y)
                            self.dx_B = abs(curr_plant_pos[0] - self.scout_B_plant_pos[1])
                            self.scout_B_current_x = curr_plant_pos[0]
                            print(f"[DX ZNALEZIONE] Scout B ({self.scout_B.name}) zmierzył odstęp: {self.dx_B:.2f} m")
                            break
                
                if self.dx_B is None:
                    self.scout_B_hopping = False

        # --- KONSENSUS LUB RATUNEK ---
        # Wystarczy, że TYLKO JEDEN łazik znajdzie DX!
        if self.dx_A is not None or self.dx_B is not None:
            
            self.search_dx_active = False # Blokujemy kolejne skoki w tym stanie
            
            if self.dx_A is not None and self.dx_B is not None:
                # Obydwa znalazły jednocześnie
                self.grid_dx = (self.dx_A + self.dx_B) / 2.0
                print(f"\n[SUKCES] OBA łaziki znalazły DX. Średnia = {self.grid_dx:.2f} m")
                
            elif self.dx_A is not None:
                # Tylko A znalazł
                self.grid_dx = self.dx_A
                self.scout_B.stop()  # Zatrzymujemy B! (Żeby nie skakał dalej w puste pole)
                print(f"\n[SUKCES] Scout A znalazł DX = {self.grid_dx:.2f} m. Zatrzymuję zgubionego Scouta B!")
                
            elif self.dx_B is not None:
                # Tylko B znalazł
                self.grid_dx = self.dx_B
                self.scout_A.stop()  # Zatrzymujemy A!
                print(f"\n[SUKCES] Scout B znalazł DX = {self.grid_dx:.2f} m. Zatrzymuję zgubionego Scouta A!")

            print("=" * 60)
            
            # Przechodzimy do rozjazdu
            self.action_in_progress = False
            self.scout_A_searching = True
            self.scout_B_searching = True
            
            # Sprzątamy zmienne pomocnicze przed zmianą stanu
            del self.dx_A
            del self.dx_B
            del self.search_dx_active
            
            self.state = "SPLIT_DIVERGE"

    def _state_find_dx(self):
        """STAN 2: Skok w prawo, aby poznać szerokość alejki (DX)."""
        if not self.action_in_progress:
            print(f"[ZWIAD] Badanie odstępu poziomego (DX)...")
            self.scout.go_to(self.anchor_pos[0] - (self.anchor_pos[1] - self.scout.pos[1]), self.scout.pos[1], heading=math.pi/2)
            self.action_in_progress = True

        if self.scout.status != "moving":
            detected = self.read_aruco_from_rover_camera(self.scout)
            for marker_id, (offset_x, offset_y) in detected.items():
                if marker_id != self.anchor_id:
                    self.scout.stop()

                    new_anchor_pos = (self.scout.pos[0] + offset_x, self.scout.pos[1] + offset_y)
                    self.grid_dx = abs(new_anchor_pos[0] - self.anchor_pos[0])
                    print(f"[SUKCES] Ustalono DX = {self.grid_dx:.2f} m")
                    
                    self.action_in_progress = False
                    self.current_edge_x = self.anchor_pos[0]
                    self.state = "FIND_REAL_ANCHOR"
                    return
                
    def _state_find_real_anchor(self):
        """STAN 3: Skoki w lewo o równe DX, aby znaleźć skrajną lewą roślinę (Prawdziwą Kotwicę)."""

        if not self.action_in_progress:
            # Planujemy skok w lewo o jedną alejkę
            target_x = self.current_edge_x - self.grid_dx
            print(f"[ZWIAD] Szukanie lewej krawędzi. Skok na pozycję X: {target_x:.2f} m...")
            
            # Jedziemy na nową pozycję po osi X, obracając kamerę z powrotem na Północ
            self.scout.go_to(target_x, self.scout.pos[1], heading=math.pi/2)
            self.action_in_progress = True

        # Gdy łazik dojedzie na miejsce i się zatrzyma
        if self.scout.status != "moving":
            detected = self.read_aruco_from_rover_camera(self.scout)
            
            if detected:
                # Widzimy roślinę! Czyli to wciąż nie jest koniec pola
                marker_id, (offset_x, offset_y) = next(iter(detected.items()))
                
                self.scout.stop()
                
                # Obliczamy i aktualizujemy nową lewą krawędź
                new_edge_pos = (self.scout.pos[0] + offset_x, self.scout.pos[1] + offset_y)
                self.current_edge_x = new_edge_pos[0]
                self.anchor_id = marker_id
                
                print(f" -> Znaleziono kolejną roślinę w lewo (ID: {marker_id}). Skaczemy dalej.")
                
                # Odblokowujemy flagę akcji – w kolejnym cyklu łazik skoczy znowu
                self.action_in_progress = False
                
            else:
                # KAMERA JEST PUSTA! Nie ma rośliny.
                # Oznacza to, że skoczyliśmy poza pole. Ostatnia znana pozycja to krawędź.
                self.scout.stop()
                
                # Przypisujemy prawdziwą skrajną lewą krawędź (Origin X)
                self.x_min = self.current_edge_x
                
                # Aktualizujemy Kotwicę, aby na niej bazował stan FIND_DY
                self.anchor_pos = (self.x_min, self.anchor_pos[1])
                
                print("\n" + "*" * 50)
                print(f"[KRAWĘDŹ ZNALEZIONA] Skrajny lewy rząd to X: {self.x_min:.2f} m")
                print("*" * 50)
                
                self.action_in_progress = False
                self.state = "FIND_DY"

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
        


    