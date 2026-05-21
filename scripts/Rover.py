"""
Rover.py — self-contained rover controller for CoppeliaSim.

All instances share Rover._class_lock (RLock) so background threads
serialize every sim API call automatically — safe for multi-rover use.
"""

import math
import time
import threading


def _normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Rover:
    # Shared reentrant lock — serializes sim calls across ALL instances
    _class_lock = threading.RLock()

    # Battery drain (%/s)
    DRAIN_IDLE          = 0.05
    DRAIN_MOVING        = 0.50
    DRAIN_TASK          = 1.00
    # Battery charge (%/s)
    STATION_CHARGE_RATE = 5.0
    SOLAR_CHARGE_RATE   = 0.10   # panels open, not at station

    # Navigation
    ARRIVE_DIST    = 0.4
    HEADING_THRESH = 0.05
    MAX_SPEED      = 10.0
    MAX_TURN       = 6.0
    WARN_DIST      = 3.5
    STOP_DIST      = 1.8

    # Clockwise routing ring radius around base
    _RING_RADIUS = 5.5

    _WHEEL_NAMES = {
        "front_left_wheel_joint":  "left",
        "rear_left_wheel_joint":   "left",
        "front_right_wheel_joint": "right",
        "rear_right_wheel_joint":  "right",
    }
    # Panel angles (from gui_control.py: open=0, closed=±π/2)
    _PANEL_OPEN   = {"left_panel_joint":  0.0,           "right_panel_joint":  0.0}
    _PANEL_CLOSED = {"left_panel_joint": -math.pi / 2,   "right_panel_joint":  math.pi / 2}

    def __init__(self, sim, handle: int, spawn_coords, charging_stations,
                 base_pos=None, panels_open: bool = False):
        """
        sim               : CoppeliaSim sim proxy
        handle            : model handle in the scene
        spawn_coords      : (x, y, z) — home position
        charging_stations : list of 4 (x,y,z) or dict {1..4: (x,y,z)}
        base_pos          : (x, y, z) of the base cube — enables clockwise routing
        panels_open       : initial panel state
        """
        self.sim      = sim
        self.handle   = handle
        self.base_pos = tuple(base_pos) if base_pos is not None else None
        self.spawn_coords = tuple(spawn_coords)

        if isinstance(charging_stations, dict):
            self.charging_stations = {int(k): tuple(v)
                                      for k, v in charging_stations.items()}
        else:
            cs = list(charging_stations)
            if len(cs) != 4:
                raise ValueError(f"Need 4 charging stations, got {len(cs)}")
            self.charging_stations = {i + 1: tuple(cs[i]) for i in range(4)}

        with Rover._class_lock:
            try:
                self.name = sim.getObjectAlias(handle, 0)
            except Exception:
                self.name = f"rover_{handle}"

        # Battery
        self._battery     = 100.0
        self._batt_lock   = threading.Lock()
        self._last_batt_t = time.time()
        self._task_active = False
        self._charging    = False   # currently receiving charge

        # Panels
        self.panels_open  = panels_open
        self._panel_joints = {}     # jname → handle

        # Navigation state
        self._target         = None
        self._target_heading = None
        self._waypoints      = []   # intermediate (x,y) waypoints before target
        self._nav_lock       = threading.RLock()
        self.status          = "idle"
        self.charging_target = None   # station ID rover is navigating to

        # Fleet awareness — set externally after all rovers are created
        self.all_rovers = []

        # Cached pose
        self.pos     = list(spawn_coords)
        self.heading = 0.0

        self._left_wheels  = []
        self._right_wheels = []
        self._init_joints()

        self._running = True
        self._thread  = threading.Thread(
            target=self._control_loop,
            name=f"rover-{self.name}",
            daemon=True,
        )
        self._thread.start()

        with Rover._class_lock:

            # create vision sensor
            cam = self.sim.createVisionSensor(
                0,
                [256, 256, 0, 0],
                [
                    0.01,                 # near clipping
                    20.0,                 # far clipping
                    math.radians(100),     # FOV
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )

            # name
            self.sim.setObjectAlias(cam, f"{self.name}_camera")

            # attach to rover
            self.sim.setObjectParent(cam, self.handle, True)

            # position relative to rover
            self.sim.setObjectPosition(
                cam,
                self.handle,
                [0.75, 0.0, 0.15]
            )

            # orientation relative to rover
            self.sim.setObjectOrientation(
                cam,
                self.handle,
                [0.0, math.pi / 2, math.pi / 2]
            )

            self.camera = cam

        # with Rover._class_lock:
        #     try:
        #         # Szukamy kamery po jej absolutnej ścieżce widocznej na Twoim screenie.
        #         # Np. dla rover_1 to będzie "/rover_1/VisionSensor"
        #         sensor_path = f"/{self.name}/visionSensor"
        #         self.camera = self.sim.getObject(sensor_path)
                
        #         print(f"[ROVER INIT] Sukces! Podpięto się pod gotowy obiektyw: {sensor_path}")
        #     except Exception as e:
        #         print(f"[ROVER INIT ERROR] Nie znaleziono 'VisionSensor' dla {self.name}: {e}")
        #         self.camera = None

    # ── Setup ──────────────────────────────────────────────────────────────

    def _init_joints(self):
        """Discover and configure wheel + panel joints."""
        with Rover._class_lock:
            all_joints = self.sim.getObjectsInTree(
                self.handle, self.sim.object_joint_type, 0)
            jmap = {self.sim.getObjectAlias(j, 0): j for j in all_joints}

        self._left_wheels  = []
        self._right_wheels = []
        for wname, side in self._WHEEL_NAMES.items():
            if wname not in jmap:
                continue
            h = jmap[wname]
            with Rover._class_lock:
                self.sim.setJointMode(h, self.sim.jointmode_force, 0)
                try:
                    self.sim.setJointTargetForce(h, 100)
                except Exception:
                    pass
            (self._left_wheels if side == "left" else self._right_wheels).append(h)

        self._panel_joints = {}
        angles = self._PANEL_OPEN if self.panels_open else self._PANEL_CLOSED
        for pname, target_angle in angles.items():
            if pname not in jmap:
                continue
            h = jmap[pname]
            with Rover._class_lock:
                self.sim.setJointMode(h, self.sim.jointmode_force, 0)
                self.sim.setJointTargetPosition(h, target_angle)
            self._panel_joints[pname] = h

    def reinit_joints(self):
        """Re-apply joint modes after sim.startSimulation()."""
        self._init_joints()

    # Backward compat alias
    reinit_wheels = reinit_joints

    # ── Control loop ───────────────────────────────────────────────────────

    def _control_loop(self):
        while self._running:
            time.sleep(0.05)
            if self.status == "dead":
                self._update_battery()
                continue
            with Rover._class_lock:
                try:
                    pos          = self.sim.getObjectPosition(self.handle, -1)
                    m            = self.sim.getObjectMatrix(self.handle, -1)
                    self.pos     = pos
                    self.heading = math.atan2(m[4], m[0])
                except Exception as e:
                    print(f"[{self.name}] pose error: {e}")
                    continue

            self._update_battery()
            if self.status == "dead":
                continue

            # When panels are open, rover stays still (solar charging)
            if self.panels_open:
                with Rover._class_lock:
                    self._apply_velocities(0.0, 0.0)
                continue

            lv, rv = self._compute_velocities()
            with Rover._class_lock:
                try:
                    self._apply_velocities(lv, rv)
                except Exception as e:
                    print(f"[{self.name}] velocity error: {e}")

    # ── Battery ────────────────────────────────────────────────────────────

    def _update_battery(self):
        now = time.time()
        dt  = now - self._last_batt_t
        self._last_batt_t = now

        at_station = (self.charging_target is not None
                      and self.status in ("arrived", "idle", "charging"))

        if at_station:
            self._charging = True
            self.status    = "charging"
            with self._batt_lock:
                self._battery = min(100.0, self._battery + self.STATION_CHARGE_RATE * dt)
            return

        if self.panels_open:
            self._charging = True
            with self._batt_lock:
                self._battery = min(100.0, self._battery + self.SOLAR_CHARGE_RATE * dt)
            return

        self._charging = False

        drain = (self.DRAIN_TASK   if self._task_active else
                 self.DRAIN_MOVING if self.status == "moving" else
                 self.DRAIN_IDLE)

        with self._batt_lock:
            self._battery = max(0.0, self._battery - drain * dt)
            depleted = self._battery == 0.0

        if depleted:
            print(f"[{self.name}] battery depleted — auto-deploying solar panels")
            self.open_panels()
            self.status = "dead"
            with Rover._class_lock:
                self._apply_velocities(0.0, 0.0)

    def set_task_active(self, active: bool):
        self._task_active = bool(active)

    @property
    def battery(self) -> float:
        with self._batt_lock:
            return round(self._battery, 1)

    @property
    def charging(self) -> bool:
        return self._charging

    # ── Panel control ──────────────────────────────────────────────────────

    def open_panels(self):
        """Deploy solar panels — rover stops moving and starts slow charging."""
        if not self._panel_joints:
            print(f"[{self.name}] no panel joints")
            return
        # Stop current movement
        with self._nav_lock:
            self._target         = None
            self._target_heading = None
            self._waypoints      = []
        with Rover._class_lock:
            self._apply_velocities(0.0, 0.0)
            for pname, h in self._panel_joints.items():
                self.sim.setJointTargetPosition(h, self._PANEL_OPEN[pname])
        self.panels_open = True
        self.status      = "idle"

    def close_panels(self):
        """Stow solar panels — charging stops, rover can move again."""
        with Rover._class_lock:
            for pname, h in self._panel_joints.items():
                self.sim.setJointTargetPosition(h, self._PANEL_CLOSED[pname])
        self.panels_open  = False
        self._charging    = False

    # ── Clockwise routing ──────────────────────────────────────────────────

    def _clockwise_path_to_station(self, station_id: int) -> list:
        """
        Return [(x,y), ...] waypoints: clockwise around the base cube to the
        target charging station.  Falls back to direct approach if base_pos
        is unknown.
        """
        station_pos = self.charging_stations[station_id][:2]

        if self.base_pos is None:
            return [station_pos]

        bx, by = self.base_pos[0], self.base_pos[1]
        R  = self._RING_RADIUS
        s  = math.sqrt(2) / 2   # sin/cos 45° ≈ 0.707

        # 8 ring waypoints in clockwise order (0=E, increasing CW)
        ring = [
            (bx + R,      by     ),   # 0  E   → station 1
            (bx + R * s,  by - R * s),# 1  SE
            (bx,          by - R ),   # 2  S   → station 4
            (bx - R * s,  by - R * s),# 3  SW
            (bx - R,      by     ),   # 4  W   → station 2
            (bx - R * s,  by + R * s),# 5  NW
            (bx,          by + R ),   # 6  N   → station 3
            (bx + R * s,  by + R * s),# 7  NE
        ]
        # Which ring index leads to each station
        station_ring = {1: 0, 4: 2, 2: 4, 3: 6}
        target_idx   = station_ring.get(station_id)
        if target_idx is None:
            return [station_pos]

        # Rover's angle from base (0-360° CCW from East)
        dx    = self.pos[0] - bx
        dy    = self.pos[1] - by
        adeg  = math.degrees(math.atan2(dy, dx)) % 360
        # First ring waypoint encountered going CW (decreasing angle)
        start_angle = math.floor(adeg / 45) * 45
        start_idx   = int((360 - start_angle) / 45) % 8

        # Walk CW from start_idx to target_idx (inclusive)
        waypoints = []
        idx = start_idx
        for _ in range(9):   # max 8 steps + 1 to include target
            waypoints.append(ring[idx])
            if idx == target_idx:
                break
            idx = (idx + 1) % 8

        waypoints.append(station_pos)
        return waypoints

    # ── Navigation ─────────────────────────────────────────────────────────

    def _compute_velocities(self):
        with self._nav_lock:
            target         = self._target
            target_heading = self._target_heading

        if target is None:
            return 0.0, 0.0

        my_x, my_y = self.pos[0], self.pos[1]
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        # Rover-position collision avoidance
        nearest_front  = None
        avoidance_turn = 0.0
        for other in self.all_rovers:
            if other is self:
                continue
            dx   = other.pos[0] - my_x
            dy   = other.pos[1] - my_y
            dist = math.hypot(dx, dy)
            if dist < 0.01:
                continue
            fwd = dx * cos_h + dy * sin_h
            if fwd <= 0 or dist >= self.WARN_DIST:
                continue
            lat   = -dx * sin_h + dy * cos_h
            theta = math.atan2(lat, fwd)
            weight = 1.0 - dist / self.WARN_DIST
            avoidance_turn -= weight * math.sin(theta) * self.MAX_TURN * 2.0
            if nearest_front is None or dist < nearest_front:
                nearest_front = dist

        if nearest_front is not None and nearest_front < self.STOP_DIST:
            if self.status == "moving":
                self.status = "blocked"
            return 0.0, 0.0

        if self.status == "blocked":
            self.status = "moving"

        # Navigate toward current target
        tx, ty = target
        dx   = tx - my_x
        dy   = ty - my_y
        dist = math.hypot(dx, dy)

        if dist < self.ARRIVE_DIST:
            with self._nav_lock:
                if self._waypoints:
                    # Advance to next waypoint
                    nxt = self._waypoints.pop(0)
                    self._target = nxt
                    tx, ty = nxt
                    dx = tx - my_x
                    dy = ty - my_y
                    dist = math.hypot(dx, dy)
                else:
                    # Final destination reached
                    if target_heading is not None:
                        err = _normalize_angle(target_heading - self.heading)
                        if abs(err) > self.HEADING_THRESH:
                            turn = min(self.MAX_TURN, abs(err) * 2.0)
                            sign = 1 if err > 0 else -1
                            self.status = "moving"
                            return sign * turn, -sign * turn
                    self._target         = None
                    self._target_heading = None
                    self.status = "arrived"
                    return 0.0, 0.0

        target_angle = math.atan2(dy, dx)
        angle_error  = _normalize_angle(target_angle - self.heading)
        speed        = min(self.MAX_SPEED, dist * 1.5)
        forward      = -speed * max(0.0, math.cos(angle_error))
        turn         = self.MAX_TURN * math.sin(angle_error) + avoidance_turn

        if nearest_front is not None and nearest_front < self.WARN_DIST:
            t = (nearest_front - self.STOP_DIST) / (self.WARN_DIST - self.STOP_DIST)
            forward *= max(0.3, min(1.0, t))

        self.status = "moving"
        left  = max(-self.MAX_SPEED, min(self.MAX_SPEED, forward + turn))
        right = max(-self.MAX_SPEED, min(self.MAX_SPEED, forward - turn))
        return left, right

    def _apply_velocities(self, lv: float, rv: float):
        for h in self._left_wheels:
            self.sim.setJointTargetVelocity(h, lv)
        for h in self._right_wheels:
            self.sim.setJointTargetVelocity(h, rv)

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def route(self):
        """Current remaining path: [(x,y), ...] from next waypoint to final target."""
        with self._nav_lock:
            if self._target is None:
                return []
            pts = [self._target]
            pts.extend(self._waypoints)
            return list(pts)

    def go_to(self, x: float, y: float, heading: float = None):
        """Navigate to (x, y). Blocked when panels are open."""
        if self.panels_open:
            print(f"[{self.name}] panels open — close before moving")
            return
        if self.status == "dead":
            return
        self.charging_target = None
        with self._nav_lock:
            self._target         = (float(x), float(y))
            self._target_heading = float(heading) if heading is not None else None
            self._waypoints      = []
            self.status          = "moving"

    def go_to_base(self):
        """Return to spawn / home position."""
        if self.panels_open:
            print(f"[{self.name}] panels open — close before moving")
            return
        if self.status == "dead":
            return
        self.charging_target = None
        x, y = self.spawn_coords[0], self.spawn_coords[1]
        with self._nav_lock:
            self._target         = (x, y)
            self._target_heading = None
            self._waypoints      = []
            self.status          = "moving"

    def go_to_charging_station(self, station_id: int):
        """
        Navigate to charging station 1–4 via clockwise route around the base.
        Auto-charges on arrival; charging stops when rover leaves.
        """
        if station_id not in self.charging_stations:
            raise ValueError(f"Unknown station_id={station_id}. "
                             f"Available: {sorted(self.charging_stations)}")
        if self.panels_open:
            print(f"[{self.name}] panels open — close before moving")
            return
        if self.status == "dead":
            return

        waypoints = self._clockwise_path_to_station(station_id)
        # waypoints[-1] is the station; everything before it is ring waypoints
        all_wps = [(float(w[0]), float(w[1])) for w in waypoints]

        self.charging_target = station_id
        with self._nav_lock:
            self._target         = all_wps[0]
            self._target_heading = None
            self._waypoints      = all_wps[1:]
            self.status          = "moving"

    def stop(self):
        """Stop and cancel goal. Clears charging target (leaves station)."""
        self.charging_target = None
        self._charging       = False
        with self._nav_lock:
            self._target         = None
            self._target_heading = None
            self._waypoints      = []
            self.status          = "idle"
        with Rover._class_lock:
            self._apply_velocities(0.0, 0.0)

    def wait_until_arrived(self, timeout: float = 60.0) -> bool:
        """Block until arrived/idle/charging. Returns True if arrived."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.status in ("arrived", "idle", "charging"):
                return True
            if self.status == "dead":
                return False
            time.sleep(0.1)
        return False

    def shutdown(self):
        self._running = False
        self.stop()

    def __repr__(self):
        return (f"<Rover {self.name!r}  {self.battery}%  "
                f"{self.status}  ({self.pos[0]:.1f},{self.pos[1]:.1f})>")
