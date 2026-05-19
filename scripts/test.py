print("--- DIAGNOSTYKA FUNKCJI CREATETEXTURE ---")
try:
    # Pobieramy informację o funkcji bezpośrednio z systemu pomocy CoppeliaSim
    # Ta komenda zwraca dokładny opis argumentów i ich typów oczekiwanych przez ZMQ
    info = sim.getApiInfo('sim.createTexture')
    print(info)
except Exception as e:
    print(f"Nie udało się pobrać ApiInfo przez sim.getApiInfo: {e}")
    
    # Alternatywna metoda diagnostyczna - sprawdzenie parametrów przez inspekcję obiektu
    try:
        import inspect
        print("Sygnatura Pythona:", inspect.signature(sim.createTexture))
    except Exception as e2:
        print(f"Nie można sprawdzić sygnatury Pythona: {e2}")

# Sprawdźmy też, jakiego typu dane generuje funkcja pakująca
try:
    test_pack = sim.packUInt8Table([255, 0, 100])
    print(f"Typ zwracany przez packUInt8Table: {type(test_pack)}")
except Exception as e3:
    print(f"packUInt8Table nie istnieje lub zgłasza błąd: {e3}")

print("-----------------------------------------")
# Przerwijmy wykonywanie, żebyś mógł zobaczyć tylko wynik testu
import sys; sys.exit()