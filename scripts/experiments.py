"""
experiments.py — zestaw 7 testów systemu wielołazikowego.

Uruchamianie:
    python experiments.py          # menu interaktywne
    python experiments.py all      # wszystkie testy po kolei

Wyniki zapisywane są do pliku CSV (jeden wiersz = jedna próbka co 2 s)
oraz do pliku experiment_summary.txt z podsumowaniem każdego testu.
"""

import sys
import os
import time
import csv
import math
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nav_gui
from nav_gui import setup_environment, detect_base, detect_charging_stations, detect_rovers, detect_plants
from planner import Planista, RoverState, FieldState
from Rover import Rover


# ============================================================
# STAŁE
# ============================================================

SAMPLE_INTERVAL = 2.0
OUTPUT_DIR      = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# BUDOWANIE / TEARDOWN SYSTEMU
# ============================================================

def build_system(n_rovers_limit: int = None):
    """
    Tworzy scenę CoppeliaSim i inicjalizuje Planistę.
    n_rovers_limit — jeśli podane, używa tylko pierwszych N łazików (test 2).

    WAŻNE: detect_rovers() tworzy obiekt Rover (z wątkiem!) dla każdego łazika
    w scenie. Jeśli n_rovers_limit < liczba łazików w scenie, nadmiarowe wątki
    muszą być natychmiast zatrzymane — inaczej przeżyją teardown i będą
    odpytywać obiekty z usuniętej sceny, powodując błędy 'object does not exist'.
    """
    client, sim = setup_environment()

    nav_gui.sim    = sim
    nav_gui.client = client

    base_pos = detect_base()
    stations = detect_charging_stations()
    all_rovers = detect_rovers(stations, base_pos)
    plants     = detect_plants()

    if n_rovers_limit is not None and n_rovers_limit < len(all_rovers):
        # Zatrzymaj wątki nadmiarowych łazików i usuń je ze sceny
        extra = all_rovers[n_rovers_limit:]
        for rv in extra:
            rv.shutdown()
            try:
                with Rover._class_lock:
                    sim.removeModel(rv.handle)
            except Exception:
                pass
        rovers = all_rovers[:n_rovers_limit]
        # Zaktualizuj all_rovers w każdym używanym łaziku
        for rv in rovers:
            rv.all_rovers = rovers
    else:
        rovers = all_rovers

    planista = Planista(
        rovers=rovers,
        plants=plants,
        charging_stations=stations,
        base_pos=base_pos,
    )
    planista.start()

    return sim, planista, rovers, plants


def teardown(sim, planista, rovers):
    """
    Poprawna kolejność zamykania:
    1. Zatrzymaj wątki łazików (shutdown) — bez tego kręcą się dalej
       i odpytują obiekty CoppeliaSim które już nie istnieją w nowej scenie,
       co powoduje zalew błędów 'object does not exist'.
    2. Odczekaj chwilę aż wątki wyjdą z pętli.
    3. Zatrzymaj Planistę.
    4. Zatrzymaj symulację.
    5. Odczekaj aż CoppeliaSim faktycznie zatrzyma scenę przed kolejnym runem.
    """
    for rv in rovers:
        try:
            rv.shutdown()
        except Exception:
            pass

    time.sleep(0.5)

    try:
        planista.stop()
    except Exception:
        pass

    try:
        sim.stopSimulation()
    except Exception:
        pass

    time.sleep(2.0)


# ============================================================
# ZBIERANIE METRYK
# ============================================================



# ============================================================
# ZBIERANIE METRYK
# ============================================================

def snapshot(planista, elapsed: float, tag: str,
             plants=None, prev_done: int = 0, prev_elapsed: float = 0.0) -> dict:
    """Pelna probka stanu systemu."""
    rover_statuses = planista.get_rover_statuses()
    field_statuses = planista.get_field_statuses()

    batteries = [v["battery"] for v in rover_statuses.values()]
    state_counts = {s: 0 for s in ("FAILED", "IDLE", "MOVING", "WORKING", "CHARGING")}
    dead_count = blocked_count = panels_open = 0

    for v in rover_statuses.values():
        ps = v["planner_state"]
        if ps in state_counts:
            state_counts[ps] += 1
        if v["rover_status"] == "dead":
            dead_count += 1
        if v["rover_status"] == "blocked":
            blocked_count += 1
        if v["panels_open"]:
            panels_open += 1

    total_wps = 0
    dist_sum = 0.0
    moving_rvs = 0
    for ri in planista.rovers.values():
        rv = ri.rover
        route = rv.route
        total_wps += len(route)
        if route and rv.status == "moving":
            tx, ty = route[-1]
            dist_sum += math.hypot(rv.pos[0] - tx, rv.pos[1] - ty)
            moving_rvs += 1
    avg_dist = round(dist_sum / moving_rvs, 1) if moving_rvs else 0.0

    done_fields     = sum(1 for v in field_statuses.values() if v["state"] == "DONE")
    occupied_fields = sum(1 for v in field_statuses.values() if v["state"] == "OCCUPIED")
    waiting_fields  = sum(1 for v in field_statuses.values() if v["state"] == "WAITING_FOR_MEASUREMENT")

    with planista.lock:
        queue_len          = len(planista.task_queue)
        active_assignments = len(planista.active_assignments)

    dt_min     = (elapsed - prev_elapsed) / 60.0
    throughput = round((done_fields - prev_done) / dt_min, 2) if dt_min > 0 else 0.0

    avg_hum = avg_fert = avg_dens = 0.0
    critical = 0
    if plants:
        hums  = [p.humidity     for p in plants]
        ferts = [p.fertility    for p in plants]
        dens  = [p.crop_density for p in plants]
        avg_hum  = round(sum(hums)  / len(hums),  1)
        avg_fert = round(sum(ferts) / len(ferts), 1)
        avg_dens = round(sum(dens)  / len(dens),  1)
        critical = sum(1 for p in plants
                       if p.humidity < 20 or p.fertility < 20 or p.crop_density < 20)

    return {
        "tag":                       tag,
        "elapsed_s":                 round(elapsed, 1),
        "n_rovers":                  len(rover_statuses),
        "failed":                    state_counts["FAILED"],
        "dead":                      dead_count,
        "idle":                      state_counts["IDLE"],
        "moving":                    state_counts["MOVING"],
        "working":                   state_counts["WORKING"],
        "charging":                  state_counts["CHARGING"],
        "blocked":                   blocked_count,
        "avg_battery":               round(sum(batteries) / max(1, len(batteries)), 1),
        "min_battery":               round(min(batteries, default=0.0), 1),
        "max_battery":               round(max(batteries, default=0.0), 1),
        "rovers_below_25pct":        sum(1 for b in batteries if b < 25),
        "rovers_below_10pct":        sum(1 for b in batteries if b < 10),
        "panels_open_count":         panels_open,
        "total_waypoints_remaining": total_wps,
        "avg_dist_to_target":        avg_dist,
        "done_fields":               done_fields,
        "occupied_fields":           occupied_fields,
        "waiting_fields":            waiting_fields,
        "total_fields":              len(field_statuses),
        "queue_len":                 queue_len,
        "active_assignments":        active_assignments,
        "throughput_fields_per_min": throughput,
        "avg_humidity":              avg_hum,
        "avg_fertility":             avg_fert,
        "avg_crop_density":          avg_dens,
        "plants_critical":           critical,
    }


def run_loop(planista, tag: str, duration_s: float,
             plants=None, stop_when_all_done: bool = True,
             sample_interval: float = SAMPLE_INTERVAL) -> list:
    """Petla probkujaca co sample_interval sekund przez max duration_s."""
    samples      = []
    start        = time.time()
    prev_done    = 0
    prev_elapsed = 0.0

    while True:
        elapsed = time.time() - start
        s = snapshot(planista, elapsed, tag,
                     plants=plants, prev_done=prev_done, prev_elapsed=prev_elapsed)
        samples.append(s)
        prev_done    = s["done_fields"]
        prev_elapsed = elapsed

        if (stop_when_all_done and s["done_fields"] == s["total_fields"]) \
                or elapsed >= duration_s:
            break
        time.sleep(sample_interval)

    return samples


# ============================================================
# ZAPIS I FORMATOWANIE WYNIKOW
# ============================================================

FIELDNAMES = [
    "tag", "elapsed_s",
    "n_rovers", "failed", "dead",
    "idle", "moving", "working", "charging", "blocked",
    "avg_battery", "min_battery", "max_battery",
    "rovers_below_25pct", "rovers_below_10pct",
    "panels_open_count",
    "total_waypoints_remaining", "avg_dist_to_target",
    "done_fields", "occupied_fields", "waiting_fields", "total_fields",
    "queue_len", "active_assignments",
    "throughput_fields_per_min",
    "avg_humidity", "avg_fertility", "avg_crop_density", "plants_critical",
]


def save_csv(samples: list, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(samples)
    print(f"  CSV -> {path}")


def completion_time(samples: list) -> float:
    total = samples[-1]["total_fields"] if samples else 0
    for s in samples:
        if s["done_fields"] == total:
            return s["elapsed_s"]
    return samples[-1]["elapsed_s"] if samples else 0.0


def summary_block(samples: list, label: str) -> list:
    """Zwraca liste linii z pelnym podsumowaniem testu."""
    if not samples:
        return [f"  {label}: brak danych"]

    last  = samples[-1]
    ct    = completion_time(samples)
    total = last["total_fields"]
    n     = len(samples)

    avg_batt   = round(sum(s["avg_battery"] for s in samples) / n, 1)
    min_batt   = min(s["min_battery"] for s in samples)
    blocked_ep = sum(1 for s in samples if s["blocked"] > 0)
    max_active = max(s["working"] + s["moving"] for s in samples)
    low_batt_t = sum(SAMPLE_INTERVAL for s in samples if s["rovers_below_25pct"] > 0)
    tps        = [s["throughput_fields_per_min"] for s in samples if s["elapsed_s"] > 2]
    avg_tp     = round(sum(tps) / len(tps), 2) if tps else 0.0
    max_tp     = max(tps, default=0.0)
    all_done   = last["done_fields"] == total

    suffix = "(wszystkie pola)" if all_done else f"(nie ukonczono - {last['done_fields']}/{total})"
    lines = [
        f"  {label}",
        f"    Czas zakonczenia:            {ct:.1f} s  {suffix}",
        f"    Obsluzone pola:              {last['done_fields']}/{total}",
        f"    Bateria sr. / min:           {avg_batt:.1f}% / {min_batt:.1f}%",
        f"    Maks. aktywnych lazikow:     {max_active}",
        f"    Czas z niska bateria (<25%): {low_batt_t:.0f} s",
        f"    Epizody blokowania:          {blocked_ep}",
        f"    Throughput sr. / maks.:      {avg_tp:.2f} / {max_tp:.2f} pol/min",
    ]

    if any(s["avg_humidity"] > 0 for s in samples):
        lines += [
            f"    Sr. wilgotnosc roslin:       {round(sum(s['avg_humidity'] for s in samples)/n,1)}%",
            f"    Sr. zyznosc roslin:          {round(sum(s['avg_fertility'] for s in samples)/n,1)}%",
            f"    Krytyczne rosliny (maks.):   {max(s['plants_critical'] for s in samples)}",
        ]

    return lines


def append_summary(lines: list, filename: str = "experiment_summary.txt"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"  Podsumowanie -> {path}")

def test1_normal(duration_s: int = 300):
    print("\n[TEST 1] Normalna praca systemu")
    sim, planista, rovers, plants = build_system()

    planista.add_all_fields_to_queue()
    samples = run_loop(planista, "T1_normal", duration_s, plants=plants)

    save_csv(samples, "T1_normal.csv")
    append_summary(
        ["=" * 65, "TEST 1 — normalna praca systemu",
         f"  Laziki: {len(rovers)}, pola: {len(plants)}", "-" * 65]
        + summary_block(samples, "Wyniki:")
    )

    teardown(sim, planista, rovers)
    print(f"  [TEST 1] zakończony — {samples[-1]['done_fields']}/{samples[-1]['total_fields']} pól")


# ============================================================
# TEST 2 — wpływ liczby łazików
# ============================================================

def test2_rover_count(counts=(2, 4, 6), duration_s: int = 300):
    print("\n[TEST 2] Wpływ liczby łazików")
    summary_rows = [
        "=" * 70,
        "TEST 2 — wpływ liczby łazików",
        "-" * 70,
        f"{'Łaziki':>7} | {'Czas [s]':>8} | {'Pola':>5} | {'Śr.bat%':>7} | {'Min.bat%':>8}",
        "-" * 70,
    ]
    all_samples = []

    for n in counts:
        print(f"  Uruchamiam z {n} łazikami...")
        sim, planista, rovers, plants = build_system(n_rovers_limit=n)

        planista.add_all_fields_to_queue()
        tag     = f"T2_{n}rovers"
        samples = run_loop(planista, tag, duration_s, plants=plants)
        all_samples.extend(samples)

        last  = samples[-1]
        ct    = completion_time(samples)
        avg_b = round(sum(s["avg_battery"] for s in samples) / len(samples), 1)
        min_b = min(s["min_battery"] for s in samples)

        summary_rows.append(
            f"{n:>7} | {ct:>8.1f} | {last['done_fields']:>2}/{last['total_fields']:<2} | "
            f"{avg_b:>7.1f} | {min_b:>8.1f}"
        )

        teardown(sim, planista, rovers)

    save_csv(all_samples, "T2_rover_count.csv")
    append_summary(summary_rows)
    print("  [TEST 2] zakończony")


# ============================================================
# TEST 3 — awaria K łazików
# ============================================================

def test3_rover_failure(k: int = 2, duration_s: int = 300, fail_after_s: int = 30):
    print(f"\n[TEST 3] Awaria {k} łazików (po {fail_after_s} s)")
    sim, planista, rovers, plants = build_system()

    planista.add_all_fields_to_queue()

    failed_event = threading.Event()

    def do_fail():
        time.sleep(fail_after_s)
        print(f"  [TEST 3] Symuluję awarię {k} łazików...")
        planista.fail_k_rovers(k)
        failed_event.set()

    threading.Thread(target=do_fail, daemon=True).start()
    samples = run_loop(planista, f"T3_fail{k}", duration_s, plants=plants)

    last = samples[-1]
    save_csv(samples, f"T3_failure_k{k}.csv")

    queue_after_fail = None
    for s in samples:
        if s["elapsed_s"] >= fail_after_s:
            queue_after_fail = s["queue_len"]
            break

    append_summary([
        "=" * 65,
        f"TEST 3 — awaria {k} lazikow",
        f"  Awaria po: {fail_after_s} s",
        "-" * 65,
    ] + summary_block(samples, f"Wyniki (awaria k={k}):") + [
        f"  Awaria wywołana po: {fail_after_s} s",
        f"  Kolejka zadań tuż po awarii: {queue_after_fail if queue_after_fail is not None else 'n/d'}",
        f"  Pola obsłużone mimo awarii: {last['done_fields']}/{last['total_fields']}",
        f"  Łaziki sprawne na koniec: {last['n_rovers'] - last['failed']}/{last['n_rovers']}",
    ])

    teardown(sim, planista, rovers)
    print(f"  [TEST 3] zakończony — {last['done_fields']}/{last['total_fields']} pól")


# ============================================================
# TEST 4 — rozładowanie baterii
# ============================================================

def test4_battery_depletion(duration_s: int = 200, deplete_after_s: int = 20):
    print(f"\n[TEST 4] Rozładowanie baterii (po {deplete_after_s} s)")
    sim, planista, rovers, plants = build_system()

    planista.add_all_fields_to_queue()

    target_name = list(planista.rovers.keys())[0]

    def do_deplete():
        time.sleep(deplete_after_s)
        rv = planista.rovers[target_name].rover
        print(f"  [TEST 4] Zeruję baterię łazika {target_name}...")
        with rv._batt_lock:
            rv._battery = 0.0

    threading.Thread(target=do_deplete, daemon=True).start()

    start             = time.time()
    samples           = []
    prev_done    = 0
    prev_elapsed = 0.0
    panels_opened_at  = None
    charging_state_at = None
    resumed_work_at   = None

    while True:
        elapsed    = time.time() - start
        s = snapshot(planista, elapsed, "T4_battery",
                     plants=plants, prev_done=prev_done, prev_elapsed=prev_elapsed)
        samples.append(s)
        prev_done    = s["done_fields"]
        prev_elapsed = elapsed

        ri = planista.rovers.get(target_name)
        if ri:
            rv = ri.rover
            if panels_opened_at is None and rv.panels_open:
                panels_opened_at = elapsed
                print(f"  [TEST 4] Panele PV otwarte po {elapsed:.1f} s")
            if charging_state_at is None and ri.state == RoverState.CHARGING:
                charging_state_at = elapsed
            if (charging_state_at is not None
                    and resumed_work_at is None
                    and ri.state in (RoverState.IDLE, RoverState.MOVING, RoverState.WORKING)
                    and rv.battery > 90.0):
                resumed_work_at = elapsed
                print(f"  [TEST 4] Łazik wznowił pracę po {elapsed:.1f} s (bat: {rv.battery:.0f}%)")

        if elapsed >= duration_s:
            break
        time.sleep(SAMPLE_INTERVAL)

    save_csv(samples, "T4_battery_depletion.csv")
    append_summary([
        "=" * 65,
        "TEST 4 — rozladowanie baterii",
        f"  Lazik testowy: {target_name}, bateria wyzerowana po: {deplete_after_s} s",
        "-" * 65,
    ] + summary_block(samples, "Wyniki:") + [
        f"  Łazik testowy: {target_name}",
        f"  Bateria wyzerowana po: {deplete_after_s} s",
        f"  Panele PV otwarte po: {f'{panels_opened_at:.1f} s' if panels_opened_at else 'nie wykryto'}",
        f"  Stan CHARGING po: {f'{charging_state_at:.1f} s' if charging_state_at else 'nie wykryto'}",
        f"  Powrót do pracy po: {f'{resumed_work_at:.1f} s' if resumed_work_at else 'nie wykryto (test za krótki — zwiększ duration_s)'}",
    ])

    teardown(sim, planista, rovers)
    print("  [TEST 4] zakończony")


# ============================================================
# TEST 5 — awaria mapy pól
# ============================================================

def test5_map_failure(duration_s: int = 300, fail_after_s: int = 40):
    print(f"\n[TEST 5] Awaria mapy pól (reset po {fail_after_s} s)")
    sim, planista, rovers, plants = build_system()

    planista.add_all_fields_to_queue()

    done_before_fail = [0]
    map_reset_at     = [None]

    def do_map_fail():
        time.sleep(fail_after_s)
        with planista.lock:
            done_before_fail[0] = sum(
                1 for f in planista.fields.values() if f.state == FieldState.DONE
            )
        print(f"  [TEST 5] Reset mapy (dotąd obsłużono: {done_before_fail[0]} pól)...")
        planista.reset_fields_now()
        map_reset_at[0] = time.time()

    threading.Thread(target=do_map_fail, daemon=True).start()

    start             = time.time()
    samples           = []
    prev_done    = 0
    prev_elapsed = 0.0
    rediscovery_start = None

    while True:
        elapsed = time.time() - start
        s = snapshot(planista, elapsed, "T5_map_fail",
                     plants=plants, prev_done=prev_done, prev_elapsed=prev_elapsed)

        if map_reset_at[0] and rediscovery_start is None and elapsed >= fail_after_s:
            rediscovery_start = elapsed

        samples.append(s)
        prev_done    = s["done_fields"]
        prev_elapsed = elapsed

        if s["done_fields"] == s["total_fields"] or elapsed >= duration_s:
            break
        time.sleep(SAMPLE_INTERVAL)

    last = samples[-1]

    recovery_time = None
    if rediscovery_start is not None:
        for s in samples:
            if s["elapsed_s"] >= rediscovery_start and s["done_fields"] == s["total_fields"]:
                recovery_time = round(s["elapsed_s"] - rediscovery_start, 1)
                break

    save_csv(samples, "T5_map_failure.csv")
    append_summary([
        "=" * 65,
        "TEST 5 — awaria mapy pol",
        f"  Reset mapy po: {fail_after_s} s",
        "-" * 65,
    ] + summary_block(samples, "Wyniki:") + [
        f"  Reset mapy po: {fail_after_s} s",
        f"  Pola obsłużone przed awarią: {done_before_fail[0]}",
        f"  Czas odtworzenia po resecie: {f'{recovery_time} s' if recovery_time else 'nie zakończono w czasie testu'}",
        f"  Pola obsłużone łącznie: {last['done_fields']}/{last['total_fields']}",
    ])

    teardown(sim, planista, rovers)
    print("  [TEST 5] zakończony")


# ============================================================
# TEST 6 — porównanie tras A* bez/z przeszkodami
# ============================================================

def test6_path_comparison(duration_s: int = 300):
    print("\n[TEST 6] Porównanie tras: A* bez przeszkód vs z przeszkodami")
    all_samples = []
    results     = {}

    for label, radius in [("bez_przeszkod", 0.0), ("z_przeszkodami", 3.0)]:
        print(f"  Tryb: {label} (plant_radius={radius})")
        sim, planista, rovers, plants = build_system()

        planista.astar.clear_obstacles()
        if radius > 0:
            planista.astar.plant_radius = radius
            planista.astar.add_plants(plants)

        positions_history = {rv.name: [] for rv in rovers}
        planista.add_all_fields_to_queue()

        start   = time.time()
        samples = []

        while True:
            elapsed = time.time() - start
            s = snapshot(planista, elapsed, f"T6_{label}",
                         plants=plants, prev_done=prev_done, prev_elapsed=prev_elapsed)
            samples.append(s)
            prev_done    = s["done_fields"]
            prev_elapsed = elapsed

            for rv in rovers:
                positions_history[rv.name].append(tuple(rv.pos[:2]))

            if s["done_fields"] == s["total_fields"] or elapsed >= duration_s:
                break
            time.sleep(SAMPLE_INTERVAL)

        total_dist = 0.0
        for positions in positions_history.values():
            total_dist += sum(
                math.hypot(positions[i][0] - positions[i-1][0],
                           positions[i][1] - positions[i-1][1])
                for i in range(1, len(positions))
            )

        results[label] = {
            "samples":    samples,
            "total_dist": round(total_dist, 1),
            "done":       samples[-1]["done_fields"],
            "total":      samples[-1]["total_fields"],
            "elapsed":    samples[-1]["elapsed_s"],
        }
        all_samples.extend(samples)
        teardown(sim, planista, rovers)

    save_csv(all_samples, "T6_path_comparison.csv")

    r0       = results["bez_przeszkod"]
    r1       = results["z_przeszkodami"]
    diff     = r0["total_dist"] - r1["total_dist"]
    diff_pct = round(diff / max(r0["total_dist"], 1) * 100, 1)

    append_summary([
        "=" * 70,
        "TEST 6 — porównanie tras",
        "-" * 70,
        f"{'Tryb':30s} | {'Dystans [m]':>11} | {'Czas [s]':>8} | {'Pola':>5}",
        "-" * 70,
        f"{'A* bez przeszkód':30s} | {r0['total_dist']:>11.1f} | {r0['elapsed']:>8.1f} | {r0['done']}/{r0['total']}",
        f"{'A* z przeszkodami':30s} | {r1['total_dist']:>11.1f} | {r1['elapsed']:>8.1f} | {r1['done']}/{r1['total']}",
        "-" * 70,
        f"  Różnica: {abs(diff):.1f} m ({abs(diff_pct):.1f}%)",
        f"  {'A* z przeszkodami jedzie krócej' if diff > 0 else 'Trasy zbliżone — teren bez przeszkód na trasie'}",
    ])

    print("  [TEST 6] zakończony")


# ============================================================
# TEST 7 — obciążenie Planisty
# ============================================================

def test7_planner_load(duration_s: int = 300):
    print("\n[TEST 7] Obciążenie Planisty — pełna kolejka od startu")
    sim, planista, rovers, plants = build_system()

    planista.add_all_fields_to_queue()

    start   = time.time()
    samples = []
    prev_done    = 0
    prev_elapsed = 0.0

    while True:
        elapsed = time.time() - start
        s = snapshot(planista, elapsed, "T7_load",
                     plants=plants, prev_done=prev_done, prev_elapsed=prev_elapsed)
        samples.append(s)
        prev_done    = s["done_fields"]
        prev_elapsed = elapsed

        if s["done_fields"] == s["total_fields"] or elapsed >= duration_s:
            break
        time.sleep(1.0)   # co 1 s — dokładniejszy obraz kolejki

    last = samples[-1]
    total = last["total_fields"]

    first_assign_time = next(
        (s["elapsed_s"] for s in samples if s["queue_len"] < total), None
    )
    max_concurrent = max(s["working"] + s["moving"] for s in samples)
    ct = completion_time(samples)

    save_csv(samples, "T7_planner_load.csv")
    append_summary([
        "=" * 65,
        "TEST 7 — obciazenie Planisty",
        f"  Laziki: {last['n_rovers']}, pola: {total}",
        "-" * 65,
    ] + summary_block(samples, "Wyniki:") + [
        f"  Zadania w kolejce na starcie: {samples[0]['queue_len']}",
        f"  Pierwsze przydzielenie po: {f'{first_assign_time:.1f} s' if first_assign_time else 'n/d'}",
        f"  Maks. jednoczesnych aktywnych łazików: {max_concurrent}",
        f"  Czas do obsłużenia wszystkich pól: {f'{ct:.1f} s' if last['done_fields'] == total else 'nie zakończono'}",
        f"  Łaziki: {last['n_rovers']}, pola: {total}",
    ])

    teardown(sim, planista, rovers)
    print(f"  [TEST 7] zakończony — {last['done_fields']}/{total} pól")


# ============================================================
# MENU / PUNKT WEJŚCIA
# ============================================================

TESTS = {
    "1": ("Normalna praca systemu",               lambda: test1_normal()),
    "2": ("Wpływ liczby łazików (2/4/6)",         lambda: test2_rover_count(counts=(2, 4, 6))),
    "3": ("Awaria 2 łazików",                     lambda: test3_rover_failure(k=2)),
    "4": ("Rozładowanie baterii",                  lambda: test4_battery_depletion()),
    "5": ("Awaria mapy pól",                       lambda: test5_map_failure()),
    "6": ("Porównanie tras A* bez/z przeszkodami", lambda: test6_path_comparison()),
    "7": ("Obciążenie Planisty",                   lambda: test7_planner_load()),
}


def print_menu():
    print("\n" + "=" * 50)
    print("  EKSPERYMENTY — system wielołazikowy")
    print("=" * 50)
    for k, (desc, _) in TESTS.items():
        print(f"  {k}. {desc}")
    print("  A. Wszystkie testy po kolei")
    print("  Q. Wyjście")
    print("=" * 50)


if __name__ == "__main__":
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.txt")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    if len(sys.argv) > 1 and sys.argv[1].lower() == "all":
        for k, (desc, fn) in TESTS.items():
            print(f"\n{'=' * 50}\n  Test {k}: {desc}\n{'=' * 50}")
            try:
                fn()
            except Exception as e:
                print(f"  [BŁĄD TEST {k}] {e}")
        print(f"\nWszystkie testy zakończone. Podsumowanie: {summary_path}")

    else:
        while True:
            print_menu()
            choice = input("Wybór: ").strip().upper()
            if choice == "Q":
                break
            elif choice == "A":
                for k, (desc, fn) in TESTS.items():
                    print(f"\nUruchamiam test {k}: {desc}")
                    try:
                        fn()
                    except Exception as e:
                        print(f"  [BŁĄD] {e}")
            elif choice in TESTS:
                desc, fn = TESTS[choice]
                print(f"\nUruchamiam: {desc}")
                try:
                    fn()
                except Exception as e:
                    print(f"  [BŁĄD] {e}")
            else:
                print("  Nieznany wybór.")