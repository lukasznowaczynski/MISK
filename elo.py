# Przykład połączenia w Twoim kodzie
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

# Twoja logika tutaj
sim.startSimulation()
message = "polaczenie dziala. ta wiadomosc jest w coppelia"
sim.addLog(sim.verbosity_scriptinfos, message)

# 3. Wyświetl potwierdzenie w Twoim terminalu Ubuntu
print(f"Wysłano do CoppeliaSim: {message}")