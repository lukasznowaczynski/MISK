import cv2
import numpy as np
import math

class RecoveryManager:
    def __init__(self, app):
        self.app = app
        self.current_scan_row = None
        self.last_plant_positions = {}
        self.lane_x_positions = {}
        self.maneuver_stage = None
        self.rover_offsets = {}

        self.state = "IDLE"
        self.action_in_progress = False
        self.scout = None
        
        self.anchor_pos = None
        self.grid_dy = None
        self.grid_dx = None
        self.x_min = None

    def recovery_mode(self):
        print("\n" + "=" * 60)
        print("[SYSTEM] URUCHAMIANIE PROCEDURY RECOVERY")
        print("=" * 60)

        print("\n[KROK 1] Odczytywanie i zabezpieczanie pozycji roślin przed awarią:")
        print("-" * 50)
        for plant in self.app.plants:
            print(f"Roslina: {plant.name} -> Rzeczywiste Globalne X: {plant.pos[0]:.2f}, Y: {plant.pos[1]:.2f}")
        print("-" * 50)
        print(f"[SUKCES] Zabezpieczono dane {len(self.app.plants)} roślin.")

        print("\n[KROK 2] Czyszczenie bazy mapowania... Symulacja amnezji floty.")
        print("[OK] Centralna mapa została wyczyszczona. Brak danych wejściowych.")

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
        
        self.app.root.after(1000, self._wait_for_base_arrival)

    def _wait_for_base_arrival(self):
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
        elif self.state == "SNAKE_SCAN":
            self._state_snake_scan()
        elif self.state == "RECOVERY_FINISHED":
            print("\n[SUKCES] Faza pełnego zwiadu wężykiem zakończona! Flota posiada kompletną mapę.")
            return

        self.app.root.after(300, self._state_machine_loop)

    def _deploy_scout(self):
        print("\n" + "=" * 60)
        print("[SYSTEM] START MAPOWANIA AWARYJNEGO")
        print("=" * 60)

        self.app.emergency_map = {}
        self.scout_A = self.app.rovers[0]
        self.scout_B = self.app.rovers[1]
        
        self.state = "DUAL_ADVANCE"
        self.action_in_progress = False

    def read_aruco_from_rover_camera(self, rv, camera_name="front"):
        with rv._class_lock:
            try:
                if camera_name == "front":
                    camera_handle = getattr(rv, 'camera', None)
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
                    fov_rad = rv.sim.getObjectFloatParam(camera_handle, rv.sim.visionfloatparam_perspective_angle)
                    cx = width / 2.0
                    cy = height / 2.0
                    focal_length = cx / math.tan(fov_rad / 2.0)
                    
                    camera_matrix = np.array([
                        [focal_length, 0, cx],
                        [0, focal_length, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    
                    dist_coeffs = np.zeros((4, 1))

                    # Fizyczny rozmiar markera (0.3m)
                    marker_size = 0.3 * (392.0 / 512.0)
                    obj_points = np.array([
                        [-marker_size/2,  marker_size/2, 0],
                        [ marker_size/2,  marker_size/2, 0],
                        [ marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)

                    for i in range(len(ids)):
                        marker_id = int(ids[i][0])
                        corner = corners[i]
                        
                        success, rvec, tvec = cv2.solvePnP(obj_points, corner, camera_matrix, dist_coeffs)
                        
                        if success:
                            offset_x = tvec[0][0]
                            offset_z = tvec[2][0] + CAMERA_OFFSET 
                            detected_data[marker_id] = (offset_x, offset_z)
                            
                return detected_data

            except Exception as e:
                print(f"[CAMERA ERROR] Błąd przetwarzania obrazu dla {rv.name}: {e}")
                return []
            
    def _local_to_global(self, rv, offset_x, offset_z, camera_name="front"):
        """Przelicza offsety kamery na globalne współrzędne mapy."""
        if camera_name == "left":
            cam_heading = rv.heading + (math.pi / 2.0)
        elif camera_name == "right":
            cam_heading = rv.heading - (math.pi / 2.0)
        else:
            cam_heading = rv.heading

        forward_x = math.cos(cam_heading)
        forward_y = math.sin(cam_heading)
        right_x = math.sin(cam_heading)
        right_y = -math.cos(cam_heading)

        g_x = rv.pos[0] + (offset_z * forward_x) + (offset_x * right_x)
        g_y = rv.pos[1] + (offset_z * forward_y) + (offset_x * right_y)

        return g_x, g_y

    def _state_dual_advance(self):
        if not self.action_in_progress:
            self.scout_A.go_to(self.scout_A.pos[0], self.scout_A.pos[1] + 50.0, heading=math.pi/2)
            self.scout_B.go_to(self.scout_B.pos[0], self.scout_B.pos[1] + 50.0, heading=math.pi/2)
            
            self.action_in_progress = True
            self.scout_A_plant_pos = None
            self.scout_B_plant_pos = None

        if not self.scout_A_plant_pos:
            det_A = self.read_aruco_from_rover_camera(self.scout_A)
            if det_A:
                marker_id, (offset_x, offset_y) = next(iter(det_A.items()))
                if offset_y <= 15.0:
                    self.scout_A.stop()
                    g_x = self.scout_A.pos[0] + offset_x
                    g_y = self.scout_A.pos[1] + offset_y
                    self.scout_A_plant_pos = (marker_id, g_x, g_y)
                    print(f"[ZWIAD A] {self.scout_A.name} wykrył roślinę, ID: {marker_id} (X={g_x:.2f}, Y={g_y:.2f})")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)

        if not self.scout_B_plant_pos:
            det_B = self.read_aruco_from_rover_camera(self.scout_B)
            if det_B:
                marker_id, (offset_x, offset_y) = next(iter(det_B.items()))
                if offset_y <= 15.0:
                    self.scout_B.stop()
                    g_x = self.scout_B.pos[0] + offset_x
                    g_y = self.scout_B.pos[1] + offset_y
                    self.scout_B_plant_pos = (marker_id, g_x, g_y)
                    print(f"[ZWIAD B] {self.scout_B.name} wykrył roślinę, ID: {marker_id} (X={g_x:.2f}, Y={g_y:.2f})")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)

        if self.scout_A_plant_pos and self.scout_B_plant_pos:
            self.action_in_progress = False
            self.state = "SPLIT_DIVERGE"

    def _state_split_diverge(self):
        if not hasattr(self, 'dx_A'):
            self.dx_A = None
            self.dx_B = None
            self.hop_dist = 6.0
            self.scout_A_hopping = False
            self.scout_B_hopping = False
            
            self.scout_A_current_x = self.scout_A_plant_pos[1]
            self.scout_B_current_x = self.scout_B_plant_pos[1]
            self.search_dx_active = True

            self.scout_A_blank_seen = False
            self.scout_B_blank_seen = False
            
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
                            g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, "right")
                            if marker_id not in self.app.emergency_map:
                                self.app.emergency_map[marker_id] = (g_x, g_y)
                            self.dx_A = abs(g_x - self.scout_A_plant_pos[1])
                            self.scout_A_current_x = g_x 
                            print(f"[DX] Scout A zmierzył odstęp: {self.dx_A:.2f} m")
                            break
                else:
                    self.scout_A_blank_seen = True
                
                if self.dx_A is None:
                    self.scout_A_hopping = False

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
                            g_x, g_y = self._local_to_global(self.scout_B, offset_x, offset_z, "left")
                            if marker_id not in self.app.emergency_map:
                                self.app.emergency_map[marker_id] = (g_x, g_y)
                            self.dx_B = abs(g_x - self.scout_B_plant_pos[1])
                            self.scout_B_current_x = g_x
                            print(f"[DX] Scout B zmierzył odstęp: {self.dx_B:.2f} m")
                            break
                else:
                    self.scout_B_blank_seen = True
                
                if self.dx_B is None:
                    self.scout_B_hopping = False

        if self.dx_A is not None or self.dx_B is not None:
            self.search_dx_active = False 
            self.scout_A_searching = True
            self.scout_B_searching = True
            
            if self.dx_A is not None and self.dx_B is not None:
                self.grid_dx = (self.dx_A + self.dx_B) / 2.0
            elif self.dx_A is not None:
                self.grid_dx = self.dx_A
                self.scout_B.stop() 
                if self.scout_B_blank_seen:
                    self.x_max = self.scout_B_plant_pos[1]
                    self.scout_B_searching = False 
            elif self.dx_B is not None:
                self.grid_dx = self.dx_B
                self.scout_A.stop() 
                if self.scout_A_blank_seen:
                    self.x_min = self.scout_A_plant_pos[1]
                    self.scout_A_searching = False 

            self.action_in_progress = False
            
            del self.dx_A
            del self.dx_B
            del self.search_dx_active
            
            self.state = "FIND_EDGES"
            
    def _state_find_edges(self):
        if not getattr(self, 'scout_A_searching', True) and not getattr(self, 'scout_B_searching', True):
            print("\n" + "=" * 60)
            print("[SUKCES] Wymiary pola poziome zmapowane!")
            print(f" -> Skrajnie lewy rząd:  X = {self.x_min:.2f} m")
            print(f" -> Skrajnie prawy rząd: X = {self.x_max:.2f} m")
            print(f" -> Szerokość alejek:    DX = {self.grid_dx:.2f} m")
            print("=" * 60)
            
            self.action_in_progress = False
            self.state = "DEPLOY_FLEET" 
            return

        if not self.action_in_progress:
            if self.scout_A_searching:
                self.scout_A.go_to(self.scout_A_current_x - self.grid_dx, self.scout_A.pos[1], heading=math.pi)
            if self.scout_B_searching:
                self.scout_B.go_to(self.scout_B_current_x + self.grid_dx, self.scout_B.pos[1], heading=0.0)
                
            self.action_in_progress = True

        wait_for_A = self.scout_A_searching and self.scout_A.status == "moving"
        wait_for_B = self.scout_B_searching and self.scout_B.status == "moving"

        if not wait_for_A and not wait_for_B:
            if self.scout_A_searching:
                det_A = self.read_aruco_from_rover_camera(self.scout_A, "right")
                if det_A:
                    marker_id, (offset_x, offset_z) = next(iter(det_A.items()))
                    g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, "right")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)
                    self.scout_A_current_x = g_x
                    self.action_in_progress = False 
                else:
                    self.x_min = self.scout_A_current_x
                    self.scout_A_searching = False
                    self.scout_A.go_to(self.x_min, self.scout_A.pos[1], heading=math.pi)
                    self.scout_A.stop()
                    self.action_in_progress = False

            if self.scout_B_searching:
                det_B = self.read_aruco_from_rover_camera(self.scout_B, "left")
                if det_B:
                    marker_id, (offset_x, offset_z) = next(iter(det_B.items()))
                    g_x, g_y = self._local_to_global(self.scout_B, offset_x, offset_z, "left")
                    if marker_id not in self.app.emergency_map:
                        self.app.emergency_map[marker_id] = (g_x, g_y)
                    self.scout_B_current_x = g_x
                    self.action_in_progress = False 
                else:
                    self.x_max = self.scout_B_current_x
                    self.scout_B_searching = False
                    self.scout_B.go_to(self.x_max, self.scout_B.pos[1], heading=0.0)
                    self.scout_B.stop()
                    self.action_in_progress = False

    def _state_find_dy(self):
        if not hasattr(self, 'dy_hopping'):
            print("\n[PROBE_DY] Rozpoczynam sondowanie pionowe...")
            self.dy_hopping = False
            self.dy_hop_dist = 2.0
            self.dy_probing_active = True
            self.action_in_progress = True
            self.dy_aligned = False 

        if self.dy_probing_active and not self.dy_aligned:
            if not hasattr(self, 'dy_aligning_started'):
                if 10 in self.app.emergency_map:
                    plant_10_x, plant_10_y = self.app.emergency_map[10]
                    target_x = plant_10_x - 5.0 
                    self.scout_A.go_to(target_x, self.scout_A.pos[1], heading=self.scout_A.heading)
                    self.dy_aligning_started = True
                else:
                    self.dy_aligned = True
            elif self.scout_A.status != "moving":
                self.dy_aligned = True
                del self.dy_aligning_started
            return 

        if self.dy_probing_active and self.dy_aligned:
            if not self.dy_hopping:
                target_y = self.scout_A.pos[1] + self.dy_hop_dist
                self.scout_A.go_to(self.scout_A.pos[0], target_y, heading=math.pi/2)
                self.dy_hopping = True
                
            elif self.scout_A.status != "moving":
                det = self.read_aruco_from_rover_camera(self.scout_A, camera_name="front")
                if det:
                    for marker_id, (offset_x, offset_z) in det.items():
                        if marker_id != self.scout_A_plant_pos[0]:
                            g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, "front")
                            self.grid_dy = abs(g_y - self.scout_A_plant_pos[2])
                            print(f"[SUKCES] Wykryto DY = {self.grid_dy:.2f} m")

                            self.new_row_start_x = g_x
                            self.new_row_start_y = g_y
                            self.dy_probing_active = False
                            break
                            
                    if self.dy_probing_active:
                        self.dy_hopping = False
                else:
                    self.dy_hopping = False

        if not self.dy_probing_active:
            self.scout_A.stop()
            self.action_in_progress = False
            
            del self.dy_hopping
            del self.dy_hop_dist
            del self.dy_probing_active
            del self.dy_aligned
            
            self.state = "SNAKE_SCAN"

    def _state_snake_scan(self):
        if not hasattr(self, 'snake_phase'):
            print("\n[SNAKE SCAN] Mapowanie pola wężykiem...")
            self.corridor_y = self.new_row_start_y - (self.grid_dy / 2.0)
            self.current_scan_x = self.x_min
            self.scan_direction = 1 
            self.snake_phase = "ALIGN_Y" 
            self.scan_hopping = False
            self.action_in_progress = True
            self.plants_found_in_current_corridor = 0

        if self.snake_phase == "ALIGN_Y":
            if not self.scan_hopping:
                self.scout_A.go_to(self.scout_A.pos[0], self.corridor_y, heading=math.pi/2)
                self.scan_hopping = True
            elif self.scout_A.status != "moving":
                self.snake_phase = "ALIGN_X"
                self.scan_hopping = False

        elif self.snake_phase == "ALIGN_X":
            if not self.scan_hopping:
                target_heading = 0.0 if self.scan_direction == 1 else math.pi
                self.scout_A.go_to(self.current_scan_x, self.corridor_y, heading=target_heading)
                self.scan_hopping = True
            elif self.scout_A.status != "moving":
                self.snake_phase = "SCAN_STEP"
                self.scan_hopping = False

        elif self.snake_phase == "SCAN_STEP":
            if not self.scan_hopping:
                for cam in ["left", "right"]:
                    det = self.read_aruco_from_rover_camera(self.scout_A, camera_name=cam)
                    if det:
                        for marker_id, (offset_x, offset_z) in det.items():
                            if marker_id not in self.app.emergency_map:
                                g_x, g_y = self._local_to_global(self.scout_A, offset_x, offset_z, cam)
                                self.app.emergency_map[marker_id] = (g_x, g_y)
                                print(f"[MAPA] Roślina ID: {marker_id:2d} (X: {g_x:.2f}, Y: {g_y:.2f})")
                                self.plants_found_in_current_corridor += 1
                
                reached_edge = False
                if self.scan_direction == 1:
                    if self.current_scan_x >= self.x_max - 0.1:
                        reached_edge = True
                    else:
                        self.current_scan_x += self.grid_dx
                else:
                    if self.current_scan_x <= self.x_min + 0.1:
                        reached_edge = True
                    else:
                        self.current_scan_x -= self.grid_dx
                        
                if reached_edge:
                    if self.plants_found_in_current_corridor == 0:
                        self.snake_phase = "DONE"
                    else:
                        self.corridor_y += self.grid_dy
                        self.scan_direction *= -1 
                        self.plants_found_in_current_corridor = 0
                        self.snake_phase = "ALIGN_Y"
                else:
                    target_heading = 0.0 if self.scan_direction == 1 else math.pi
                    self.scout_A.go_to(self.current_scan_x, self.corridor_y, heading=target_heading)
                    self.scan_hopping = True
                    
            elif self.scout_A.status != "moving":
                self.scan_hopping = False

        elif self.snake_phase == "DONE":
            self.scout_A.stop()
            self.action_in_progress = False
            
            print("\n" + "=" * 60)
            print(f"[SUKCES] Mapa awaryjna zrekonstruowana! Łącznie roślin: {len(self.app.emergency_map)}")
            print("=" * 60)
            
            del self.snake_phase
            del self.scan_hopping
            del self.corridor_y
            del self.current_scan_x
            del self.scan_direction
            del self.plants_found_in_current_corridor
            
            self.state = "RECOVERY_FINISHED"