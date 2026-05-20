import time
from generate_env import setup_simulation_environment
def main():
    client, sim = setup_simulation_environment()
    time.sleep(1.0)

    try:
        rover3_handle = sim.getObject('/rover_3')
        rover5_handle = sim.getObject('/rover_5')
        pos3 = sim.getObjectPosition(rover3_handle, -1)
        pos5 = sim.getObjectPosition(rover5_handle, -1)
        
        print(f"Pozycja łazika 3 (X, Y, Z): [{pos3[0]:.2f}, {pos3[1]:.2f}, {pos3[2]:.2f}]")
        print(f"Pozycja łazika 5 (X, Y, Z): [{pos5[0]:.2f}, {pos5[1]:.2f}, {pos5[2]:.2f}]")
        
    except Exception as e:
        print(f"Błąd podczas odczytywania pozycji łazików: {e}")
        
    while True:
        pos3 = sim.getObjectPosition(rover3_handle, -1)
        print(f"Bieżąca pozycja łazika 3: {pos3}")
        time.sleep(0.1)




if __name__ == "__main__":
    main()