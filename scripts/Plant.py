"""Plant.py — plant state management for CoppeliaSim rover simulation."""

import math
import random
import threading
import time

from Rover import Rover


class Plant:
    # Per-second decay ranges (min, max) — randomised per instance
    _HUMIDITY_DECAY  = (0.04, 0.12)
    _FERTILITY_DECAY = (0.02, 0.07)
    _DENSITY_DECAY   = (0.01, 0.04)

    MEASURE_DELAY      = 2.0
    WATER_DURATION     = 5.0
    FERTILIZE_DURATION = 8.0
    SOW_DURATION       = 10.0
    WATER_RESTORE      = 40.0
    FERTILIZE_RESTORE  = 30.0
    SOW_RESTORE        = 25.0
    ACTION_RANGE       = 3.0   # rover must be within this many metres to act

    def __init__(self, sim, handle: int, name: str = None):
        self.sim    = sim
        self.handle = handle
        self.name   = name or f"plant_{handle}"

        self.humidity     = random.uniform(60.0, 95.0)
        self.fertility    = random.uniform(55.0, 90.0)
        self.crop_density = random.uniform(50.0, 85.0)

        self._hum_decay  = random.uniform(*self._HUMIDITY_DECAY)
        self._fert_decay = random.uniform(*self._FERTILITY_DECAY)
        self._dens_decay = random.uniform(*self._DENSITY_DECAY)

        self._lock    = threading.Lock()
        self._pending = {}   # param → (ready_at, snapshot_value)

        self.action          = None   # None | 'watering' | 'fertilizing' | 'sowing'
        self.action_progress = 0.0
        self._action_start   = None
        self._action_end     = None

        self.pos = [0.0, 0.0, 0.0]
        with Rover._class_lock:
            try:
                self.pos = list(sim.getObjectPosition(handle, -1))
            except Exception:
                pass

        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name=f"plant-{self.name}", daemon=True)
        self._thread.start()

    # ── Background loop ────────────────────────────────────────────────────

    def _loop(self):
        last_t = time.time()
        while self._running:
            time.sleep(0.1)
            now = time.time()
            dt  = now - last_t
            last_t = now
            with self._lock:
                if self.action != 'watering':
                    self.humidity     = max(0.0, self.humidity     - self._hum_decay  * dt)
                if self.action != 'fertilizing':
                    self.fertility    = max(0.0, self.fertility    - self._fert_decay * dt)
                if self.action != 'sowing':
                    self.crop_density = max(0.0, self.crop_density - self._dens_decay * dt)
                if self.action is not None:
                    span = self._action_end - self._action_start
                    self.action_progress = min(1.0, (now - self._action_start) / span)
                    if now >= self._action_end:
                        self._finish_action()

    def _finish_action(self):
        """Called with self._lock held."""
        if self.action == 'watering':
            self.humidity     = min(100.0, self.humidity     + self.WATER_RESTORE)
        elif self.action == 'fertilizing':
            self.fertility    = min(100.0, self.fertility    + self.FERTILIZE_RESTORE)
        elif self.action == 'sowing':
            self.crop_density = min(100.0, self.crop_density + self.SOW_RESTORE)
        self.action = None
        self.action_progress = 0.0
        self._action_start = self._action_end = None

    # ── Measurement API ────────────────────────────────────────────────────

    def start_measure(self, param: str) -> bool:
        """Begin a delayed measurement. Returns False if one is already pending."""
        with self._lock:
            if param in self._pending:
                return False
            self._pending[param] = (time.time() + self.MEASURE_DELAY,
                                    getattr(self, param))
            return True

    def get_measure(self, param: str):
        """Returns ('idle'|'pending'|'ready', value_or_None). Clears entry on ready."""
        with self._lock:
            if param not in self._pending:
                return 'idle', None
            ready_at, value = self._pending[param]
            if time.time() < ready_at:
                return 'pending', None
            del self._pending[param]
            return 'ready', round(value, 1)

    def measure_all(self):
        for p in ('humidity', 'fertility', 'crop_density'):
            self.start_measure(p)

    # ── Action API ─────────────────────────────────────────────────────────

    def _start_action(self, name: str, duration: float) -> bool:
        with self._lock:
            if self.action is not None:
                return False
            now = time.time()
            self.action          = name
            self.action_progress = 0.0
            self._action_start   = now
            self._action_end     = now + duration
            return True

    def water(self)     -> bool: return self._start_action('watering',    self.WATER_DURATION)
    def fertilize(self) -> bool: return self._start_action('fertilizing', self.FERTILIZE_DURATION)
    def sow(self)       -> bool: return self._start_action('sowing',      self.SOW_DURATION)

    def rover_nearby(self, rovers: list) -> bool:
        px, py = self.pos[0], self.pos[1]
        return any(
            math.hypot(rv.pos[0] - px, rv.pos[1] - py) <= self.ACTION_RANGE
            for rv in rovers
        )

    def shutdown(self):
        self._running = False

    def __repr__(self):
        return (f"<Plant {self.name!r} "
                f"H={self.humidity:.0f}% F={self.fertility:.0f}% D={self.crop_density:.0f}%>")
