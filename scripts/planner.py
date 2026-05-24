import time
import math
import heapq
import threading
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from Rover import Rover
import random

# ============================================================
# STANY
# ============================================================

class RoverState(Enum):
    FAILED = auto()
    IDLE = auto()
    MOVING = auto()
    WORKING = auto()
    CHARGING = auto()
    REMAPPING = auto()


class FieldState(Enum):
    WAITING_FOR_MEASUREMENT = auto()
    OCCUPIED = auto()
    DONE = auto()


# ============================================================
# MODELE PLANISTY
# ============================================================

@dataclass
class FieldTask:
    field_id: str
    task_type: str = "FULL_SERVICE"


@dataclass
class FieldInfo:
    field_id: str
    plant: object
    position: Tuple[float, float]
    state: FieldState = FieldState.WAITING_FOR_MEASUREMENT
    assigned_rover: Optional[str] = None


@dataclass
class RoverInfo:
    name: str
    rover: object
    state: RoverState = RoverState.IDLE
    assigned_field: Optional[str] = None
    current_task: Optional[FieldTask] = None
    failed: bool = False


# ============================================================
# A* — NA RAZIE BEZ PRZESZKÓD
# ============================================================

class AStarPlanner:
    def __init__(self, grid_size=200, offset=100):
        self.grid_size = grid_size
        self.offset = offset


    def world_to_grid(self, pos):
        x, y = pos
        return int(round(x + self.offset)), int(round(y + self.offset))

    def grid_to_world(self, cell):
        x, y = cell
        return float(x - self.offset), float(y - self.offset)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def neighbors(self, cell):
        x, y = cell
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        return [c for c in candidates if self.in_bounds(c)]

    def plan(self, start_world, goal_world):
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.neighbors(current):
                new_cost = g_score[current] + 1

                if neighbor not in g_score or new_cost < g_score[neighbor]:
                    g_score[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current

        return []

    def reconstruct_path(self, came_from, current):
        path = []

        while current is not None:
            path.append(self.grid_to_world(current))
            current = came_from[current]

        path.reverse()
        return path


# ============================================================
# PLANISTA
# ============================================================

class Planista:
    def __init__(
        self,
        rovers,
        plants,
        charging_stations=None,
        base_pos=None,
        reset_interval_sec=30 * 60,
    ):
        self.rovers: Dict[str, RoverInfo] = {}
        self.fields: Dict[str, FieldInfo] = {}
        self.task_queue: List[FieldTask] = []

        self.charging_stations = charging_stations or {}
        self.base_pos = base_pos

        self.reset_interval_sec = reset_interval_sec
        self.last_reset_time = time.time()

        self.astar = AStarPlanner()
        self.start_delay_between_rovers = 1.5
        self.active_assignments = {}
        self.reserved_charging_stations = {}
        self.reset_generation = 0
        self.lock = threading.RLock()
        self.running = False
        self.thread = None

        for rover in rovers:
            self.rovers[rover.name] = RoverInfo(
                name=rover.name,
                rover=rover,
                state=RoverState.IDLE,
            )

        for plant in plants:
            self.fields[plant.name] = FieldInfo(
                field_id=plant.name,
                plant=plant,
                position=(plant.pos[0], plant.pos[1]),
            )

    # ========================================================
    # START / STOP
    # ========================================================

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self.main_loop, daemon=True)
        self.thread.start()
        print("[Planista] Start")

    def stop(self):
        self.running = False
        print("[Planista] Stop")

    # ========================================================
    # GŁÓWNA PĘTLA
    # ========================================================

    def main_loop(self):
        while self.running:
            with self.lock:
                self.monitor_batteries()
                self.reset_fields_if_needed()
                self.assign_tasks()

            time.sleep(1.0)

    # ========================================================
    # KOLEJKA ZADAŃ
    # ========================================================

    def add_task(self, field_id, task_type="FULL_SERVICE"):
        with self.lock:
            if field_id not in self.fields:
                return

            field = self.fields[field_id]

            if field.state != FieldState.WAITING_FOR_MEASUREMENT:
                return

            for task in self.task_queue:
                if task.field_id == field_id:
                    return

            self.task_queue.append(FieldTask(field_id, task_type))
            print(f"[Planista] Dodano zadanie: {field_id}")

    def add_all_fields_to_queue(self):
        for field_id in self.fields:
            self.add_task(field_id)

    def get_queue_snapshot(self):
        with self.lock:
            return [task.field_id for task in self.task_queue]

    # ========================================================
    # RESET PÓL CO 30 MINUT
    # ========================================================

    def reset_fields_if_needed(self):
        if time.time() - self.last_reset_time < self.reset_interval_sec:
            return

        print("[Planista] Reset pól — nowa runda pomiarowa")

        for field in self.fields.values():
            field.state = FieldState.WAITING_FOR_MEASUREMENT
            field.assigned_rover = None

        self.task_queue.clear()
        self.add_all_fields_to_queue()
        self.last_reset_time = time.time()

    def reset_fields_now(self):
        with self.lock:
            for field in self.fields.values():
                field.state = FieldState.WAITING_FOR_MEASUREMENT
                field.assigned_rover = None

            self.task_queue.clear()
            self.add_all_fields_to_queue()
            self.last_reset_time = time.time()

    # ========================================================
    # PRZYDZIAŁ ZADAŃ
    # ========================================================

    def assign_tasks(self):
        if not self.task_queue:
            return

        free_rovers = self.get_free_rovers()

        if not free_rovers:
            return

        while self.task_queue and free_rovers:
            best_rover = None
            best_task = None
            best_cost = float("inf")

            for rover_info in free_rovers:
                rover_pos = self.get_rover_position(rover_info)

                for task in self.task_queue:
                    field = self.fields[task.field_id]

                    if field.state != FieldState.WAITING_FOR_MEASUREMENT:
                        continue

                    cost = self.distance(rover_pos, field.position)

                    # lekka kara, jeśli bateria jest niższa
                    cost += max(0, 100 - rover_info.rover.battery) / 10

                    if cost < best_cost:
                        best_cost = cost
                        best_rover = rover_info
                        best_task = task

            if best_rover is None or best_task is None:
                return

            self.task_queue.remove(best_task)
            self.start_task(best_rover, best_task)

            free_rovers = self.get_free_rovers()

    def get_free_rovers(self):
        result = []

        for rover_info in self.rovers.values():
            if rover_info.failed:
                continue

            if rover_info.state != RoverState.IDLE:
                continue

            if rover_info.rover.battery < 25:
                continue

            result.append(rover_info)

        return result

    def start_task(self, rover_info, task):
        field = self.fields[task.field_id]

        field.state = FieldState.OCCUPIED
        field.assigned_rover = rover_info.name

        rover_info.state = RoverState.MOVING
        rover_info.assigned_field = field.field_id
        rover_info.current_task = task

        self.active_assignments[rover_info.name] = field.field_id

        delay = len(self.active_assignments) * self.start_delay_between_rovers

        print(f"[Planista] {rover_info.name} -> {field.field_id}, start za {delay:.1f}s")

        generation = self.reset_generation

        thread = threading.Thread(
            target=self.execute_field_task_with_delay,
            args=(rover_info, field, delay, generation),
            daemon=True,
        )
        thread.start()
    #opóźnienie
    def execute_field_task_with_delay(self, rover_info, field, delay, generation):
        time.sleep(delay)

        if generation != self.reset_generation:
            return

        self.execute_field_task(rover_info, field, generation)

    # ========================================================
    # WYKONANIE ZADANIA
    # ========================================================

    def execute_field_task(self, rover_info, field, generation):
        rover = rover_info.rover
        if generation != self.reset_generation:
            return
        try:
            # 1. Trasa A* do punktu obok roślinki
            approach_x, approach_y = self.get_approach_point(rover_info, field)

            path = self.astar.plan(
                self.get_rover_position(rover_info),
                (approach_x, approach_y),
            )

            # 2. Jazda waypointami
            for waypoint in path[::5]:
                if generation != self.reset_generation:
                    return
                if rover_info.failed:
                    self.return_task_to_queue(field)
                    return

                if rover.battery < 25:
                    self.return_task_to_queue(field)
                    rover_info.assigned_field = None
                    rover_info.current_task = None
                    self.send_rover_to_charge(rover_info)
                    return

                rover.go_to(waypoint[0], waypoint[1])
                self.wait_until_arrived(rover, timeout=8)

            # 3. Dojazd dokładnie do punktu obok roślinki
            rover.go_to(approach_x, approach_y)
            self.wait_until_arrived(rover, timeout=15)

            # 4. Praca na polu z odległości, bez wjeżdżania w roślinkę
            rover_info.state = RoverState.WORKING
            rover.set_task_active(True)

            self.perform_full_service(field)

            rover.set_task_active(False)

            # 5. Pole obsłużone
            field.state = FieldState.DONE
            field.assigned_rover = None

            rover_info.assigned_field = None
            rover_info.current_task = None

            # 6. Po zadaniu wraca do bazy
            rover_info.state = RoverState.MOVING
            self.send_rover_to_base(rover_info)

            rover_info.state = RoverState.IDLE

            print(f"[Planista] {rover_info.name} zakończył {field.field_id}")

        except Exception as e:
            print(f"[Planista] Błąd zadania {field.field_id}: {e}")
            self.return_task_to_queue(field)

            rover_info.state = RoverState.IDLE
            rover_info.assigned_field = None
            rover_info.current_task = None

        finally:
            rover.set_task_active(False)

            if rover_info.name in self.active_assignments:
                del self.active_assignments[rover_info.name]

    def get_approach_point(self, rover_info, field):
        rover_pos = self.get_rover_position(rover_info)
        fx, fy = field.position
        rx, ry = rover_pos

        dx = rx - fx
        dy = ry - fy
        dist = math.hypot(dx, dy)

        # jeśli rover jest dokładnie na pozycji pola, wybierz domyślny kierunek
        if dist < 0.01:
            dx, dy = 1.0, 0.0
            dist = 1.0

        # punkt 3 metry od rośliny od strony, z której nadjeżdża robot
        safe_distance = 3.0

        ux = dx / dist
        uy = dy / dist

        approach_x = fx + ux * safe_distance
        approach_y = fy + uy * safe_distance

        return approach_x, approach_y

    def perform_full_service(self, field):
        plant = field.plant

        plant.measure_all()
        time.sleep(2.2)

        values = {}

        for param in ("humidity", "fertility", "crop_density"):
            state, value = plant.get_measure(param)
            values[param] = value

        print(f"[Planista] Pomiar {field.field_id}: {values}")

        # Po obsłużeniu pola ustawiamy parametry na 100%
        with plant._lock:
            plant.humidity = 100.0
            plant.fertility = 100.0
            plant.crop_density = 100.0
            plant.action = None
            plant.action_progress = 0.0
            plant._action_start = None
            plant._action_end = None

        print(f"[Planista] Pole {field.field_id} obsłużone — parametry ustawione na 100%")

    def wait_until_arrived(self, rover, timeout=10):
        start = time.time()

        while time.time() - start < timeout:
            if rover.status in ("arrived", "idle", "charging"):
                return True

            time.sleep(0.2)

        return False

    # ========================================================
    # BATERIA / ŁADOWANIE
    # ========================================================

    def monitor_batteries(self):
        for rover_info in self.rovers.values():
            rover = rover_info.rover

            if rover_info.failed:
                continue

            # Bateria całkowicie padła — awaryjne ładowanie panelami
            if rover.status == "dead" or rover.battery <= 0:
                self.handle_rover_battery_depleted(rover_info)
                continue

            # Niska bateria — wysyłamy do ładowarki
            if rover.battery < 25 and rover_info.state not in (
                    RoverState.CHARGING,
                    RoverState.FAILED,
            ):
                print(f"[Planista] {rover_info.name} ma niski poziom baterii")

                if rover_info.assigned_field:
                    field = self.fields[rover_info.assigned_field]
                    self.return_task_to_queue(field)
                    rover_info.assigned_field = None
                    rover_info.current_task = None

                self.send_rover_to_charge(rover_info)
                continue

            # Ładowanie zakończone dopiero przy 95%
            if rover_info.state == RoverState.CHARGING and rover.battery >= 95:
                if rover.panels_open:
                    rover.close_panels()

                self.reserved_charging_stations.pop(rover_info.name, None)
                rover.charging_target = None
                rover_info.state = RoverState.IDLE

                print(f"[Planista] {rover_info.name} zakończył ładowanie")

    def handle_rover_battery_depleted(self, rover_info):
        rover = rover_info.rover

        print(f"[Planista] {rover_info.name} bateria krytyczna — panele PV")

        if rover_info.assigned_field:
            field = self.fields[rover_info.assigned_field]
            self.return_task_to_queue(field)

        rover.open_panels()
        rover_info.state = RoverState.CHARGING

    def send_rover_to_charge(self, rover_info):
        rover = rover_info.rover

        if not self.charging_stations:
            rover.open_panels()
            rover_info.state = RoverState.CHARGING
            return

        # stacje już zajęte albo zarezerwowane
        occupied = {
            r.rover.charging_target
            for r in self.rovers.values()
            if r.rover.charging_target is not None
        }

        occupied.update(self.reserved_charging_stations.values())

        free_stations = [
            sid for sid in self.charging_stations.keys()
            if sid not in occupied
        ]

        # jeśli wszystkie stacje zajęte, łazik czeka i ładuje się panelami
        if not free_stations:
            print(f"[Planista] Brak wolnej ładowarki dla {rover_info.name}, otwieram panele")
            rover.open_panels()
            rover_info.state = RoverState.CHARGING
            return

        rover_pos = self.get_rover_position(rover_info)

        closest_station = min(
            free_stations,
            key=lambda sid: self.distance(
                rover_pos,
                (self.charging_stations[sid][0], self.charging_stations[sid][1])
            )
        )

        self.reserved_charging_stations[rover_info.name] = closest_station

        print(f"[Planista] {rover_info.name} jedzie do ładowarki {closest_station}")

        rover.go_to_charging_station(closest_station)
        rover_info.state = RoverState.CHARGING

    def send_rover_to_base(self, rover_info):
        if self.base_pos is None:
            return

        rover_info.rover.go_to_base()
        self.wait_until_arrived(rover_info.rover, timeout=20)

    # ========================================================
    # AWARIE
    # ========================================================

    def fail_rover(self, rover_name):
        if rover_name not in self.rovers:
            return

        rover_info = self.rovers[rover_name]

        print(f"[Planista] Awaria łazika {rover_name}")

        rover_info.failed = True
        rover_info.state = RoverState.FAILED

        try:
            rover_info.rover.stop()
        except Exception:
            pass

        if rover_info.assigned_field:
            field = self.fields[rover_info.assigned_field]
            self.return_task_to_queue(field)

        rover_info.assigned_field = None
        rover_info.current_task = None

    def fail_k_rovers(self, k):
        count = 0

        for rover_name in list(self.rovers.keys()):
            if count >= k:
                break

            rover_info = self.rovers[rover_name]

            if not rover_info.failed:
                self.fail_rover(rover_name)
                count += 1

    def return_task_to_queue(self, field):
        field.state = FieldState.WAITING_FOR_MEASUREMENT
        field.assigned_rover = None

        exists = any(task.field_id == field.field_id for task in self.task_queue)

        if not exists:
            self.task_queue.insert(0, FieldTask(field.field_id))

    # ========================================================
    # SCENARIUSZE
    # ========================================================

    def scenario_all_rovers_ok(self):
        print("[Planista] Scenariusz 1")
        self.reset_fields_now()

    def scenario_k_rovers_failure(self, k=2):
        print("[Planista] Scenariusz 2")
        self.reset_fields_now()

        threading.Timer(5.0, lambda: self.fail_k_rovers(k)).start()

    def scenario_battery_depletion(self):
        print("[Planista] Scenariusz 3")

        for rover_info in self.rovers.values():
            if not rover_info.failed:
                rover_info.rover._battery = 0.0
                self.handle_rover_battery_depleted(rover_info)
                return

    # ========================================================
    # STATUSY DO GUI
    # ========================================================

    def get_rover_statuses(self):
        data = {}

        for name, rover_info in self.rovers.items():
            rover = rover_info.rover

            data[name] = {
                "battery": rover.battery,
                "planner_state": rover_info.state.name,
                "rover_status": rover.status,
                "position": tuple(rover.pos[:2]),
                "base_relative_position": self.get_relative_to_base(rover_info),
                "panels_open": rover.panels_open,
                "assigned_field": rover_info.assigned_field,
            }

        return data

    def get_field_statuses(self):
        return {
            field_id: {
                "state": field.state.name,
                "assigned_rover": field.assigned_rover,
                "position": field.position,
            }
            for field_id, field in self.fields.items()
        }

    def get_relative_to_base(self, rover_info):
        if self.base_pos is None:
            return None

        x = rover_info.rover.pos[0] - self.base_pos[0]
        y = rover_info.rover.pos[1] - self.base_pos[1]

        return x, y
    # ========================================================
    # POMOCNICZE
    # ========================================================
    #reset łazików do pozycji startowych
    def reset_rovers_to_start(self):
        print("[Planista] Reset łazików do pozycji startowych")

        with self.lock:
            self.reset_generation += 1

            self.task_queue.clear()
            self.active_assignments.clear()
            self.reserved_charging_stations.clear()

            for field in self.fields.values():
                field.state = FieldState.WAITING_FOR_MEASUREMENT
                field.assigned_rover = None

            for rover_info in self.rovers.values():
                rover = rover_info.rover

                try:
                    # zatrzymanie aktualnego ruchu i trasy
                    rover.stop()

                    if hasattr(rover, "_nav_lock"):
                        with rover._nav_lock:
                            rover._target = None
                            rover._target_heading = None
                            rover._waypoints = []

                    # zamknięcie paneli
                    if rover.panels_open:
                        rover.close_panels()

                    # reset baterii
                    if hasattr(rover, "_batt_lock"):
                        with rover._batt_lock:
                            rover._battery = 100.0

                    # reset flag ładowania
                    rover.charging_target = None
                    if hasattr(rover, "_charging"):
                        rover._charging = False

                    # ustawienie pozycji startowej
                    with Rover._class_lock:
                        rover.sim.setObjectPosition(
                            rover.handle,
                            -1,
                            list(rover.spawn_coords)
                        )
                        rover.sim.setObjectOrientation(
                            rover.handle,
                            -1,
                            [0, 0, math.radians(90)]
                        )

                    # aktualizacja lokalnej pozycji
                    rover.pos = list(rover.spawn_coords)
                    rover.heading = math.radians(90)
                    rover.status = "idle"

                    # reset stanu Planisty
                    rover_info.state = RoverState.IDLE
                    rover_info.failed = False
                    rover_info.assigned_field = None
                    rover_info.current_task = None

                except Exception as e:
                    print(f"[Planista] Błąd resetu {rover_info.name}: {e}")

        print("[Planista] Reset zakończony")
    #lepszy opis kolejki
    def get_queue_details(self):
        with self.lock:
            rows = []

            for task in self.task_queue:
                rows.append(f"OCZEKUJE: {task.field_id} | zadanie: {task.task_type}")

            for rover_name, field_id in self.active_assignments.items():
                rows.append(f"W TRAKCIE: {rover_name} obsługuje {field_id}")

            for rover_info in self.rovers.values():
                if rover_info.state == RoverState.CHARGING:
                    rows.append(f"ŁADOWANIE: {rover_info.name} | bateria {rover_info.rover.battery}%")

                if rover_info.state == RoverState.FAILED:
                    rows.append(f"AWARIA: {rover_info.name}")

            return rows
    def get_rover_position(self, rover_info):
        return rover_info.rover.pos[0], rover_info.rover.pos[1]

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])