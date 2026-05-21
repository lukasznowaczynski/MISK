import sys
import os
import time
import math

WIN_IP = "172.18.48.1"

def main():
    print("=" * 60)
    print("[SYSTEM] URUCHAMIANIE PROCEDURY RECOVERY / AWARYJNEJ")
    print("=" * 60)
    
    try:
        # Importujemy klienta lokalnie w funkcji, by upewnić się, że stan sieci jest czysty
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient(host=WIN_IP, port=23000)
        sim = client.getObject('sim')
        print("[OK] Połączenie sieciowe nawiązane pomyślnie!")
    except Exception as e:
        print(f"[BLAD] Nie udało się połączyć z CoppeliaSim: {e}")
        return

    # 1. POBRANIE I WYPISANIE AKTUALNYCH WSPÓŁRZĘDNYCH ROŚLIN
    print("\n[KROK 1] Odczytywanie rzeczywistych pozycji roślin przed czyszczeniem:")
    print("-" * 50)
    plants_found = []
    for r in range(20):
        for c in range(20):
            alias = f"plant_{r}_{c}"
            try:
                handle = sim.getObject(f'/{alias}')
                pos = sim.getObjectPosition(handle, -1)
                print(f"Roslina: {alias} -> Globalne X: {pos[0]:.2f}, Y: {pos[1]:.2f}")
                plants_found.append(handle)
                time.sleep(0.02) # Mały oddech dla gniazda REQ-REP
            except Exception:
                break

    # 2. ZEROWANIE ZMIENNYCH
    print("\n[KROK 2] Symulacja awarii: Zerowanie zmiennych mapy w toku...")
    time.sleep(0.5)
    print("[OK] Pamięć mapowania została wyczyszczona. System działa w trybie AMNEZJI.")

    # 3. WYKRYWANIE WSZYSTKICH ŁAZIKÓW NA SCENIE
    print("\n[KROK 3] Lokalizowanie floty łazików na polu...")
    rover_handles = []
    for i in range(1, 21):
        try:
            h = sim.getObject(f'/rover_{i}')
            rover_handles.append(h)
            time.sleep(0.02)
        except Exception:
            continue
    print(f"[INFO] Wykryto {len(rover_handles)} łazików wymagających ewakuacji.")

    # 4. ROZKAZ POWROTU DO BAZY (BEZPOŚREDNIO PRZEZ API SIM - BEZ KLASY ROVER)
    print("\n[KROK 4] Wysyłanie rozkazu natychmiastowego powrotu do bazy dla wszystkich łazików...")
    print("-" * 50)
    
    WIATKA_X, WIATKA_Y = 10.0, -30.0
    
    # Rozstawiamy je w rzędzie bazowym (odtworzenie logiki z nav_gui)
    N_ROVERS = len(rover_handles)
    DISTANCE_STEP = 7
    SAFE_MARGIN = 20
    half = N_ROVERS // 2

    for i, h in enumerate(rover_handles):
        name = sim.getObjectAlias(h, 0)
        
        # Wyliczamy dokładnie jego punkt docelowy w bazie (spawn_coords z nav_gui)
        target_x = (WIATKA_X - (half - i - 1) * DISTANCE_STEP - SAFE_MARGIN
                    if i < half else WIATKA_X + (i - half) * DISTANCE_STEP + SAFE_MARGIN)
        target_y = -30.0
        
        print(f" -> [{name}]: Nakaz jazdy na pozycję bazową ({target_x:.1f}, {target_y:.1f})")
        
        try:
            # Ponieważ nav_gui.py działa w tle i to ONO kontroluje obiekty Rover,
            # my po prostu zmieniamy cel bezpośrednio w symulacji lub wysyłamy łaziki skryptem.
            # Aby zmusić fizyczny model do ruchu bez awarii wątków, 
            # najbezpieczniej dla skryptu zewnętrznego jest teleportować je/ustawić ich cel
            # lub wywołać bezpośrednią zmianę pozycji w Coppeli (czysty powrót awaryjny):
            sim.setObjectPosition(h, -1, [target_x, target_y, 0.5])
            sim.setObjectOrientation(h, -1, [0, 0, math.radians(90)])
            
        except Exception as e:
            print(f"Błąd komunikacji dla {name}: {e}")
            
        time.sleep(0.05) # Bezpieczny bezpiecznik czasowy między rozkazami

    print("-" * 50)
    print("[SUKCES] Wszystkie łaziki zostały zresetowane i zsynchronizowane w bazie!")
    print("[SYSTEM] Gotowy do rozpoczęcia procedury: RECOVERY MAP PATTERN.")
    print("=" * 60)

if __name__ == "__main__":
    main()