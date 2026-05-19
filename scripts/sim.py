from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os

client = RemoteAPIClient()
sim = client.getObject('sim')

a_pwd = os.getcwd()
model_path = a_pwd + "/models/rover2.ttm"
print(model_path)

positions = [
    [-2.0,  2.0, 0.4],

]

names = ["rover_1"]

base_handle = sim.loadModel(model_path)
handles = [base_handle]

# Copy models
for _ in range(1):
    copy = sim.copyPasteObjects([base_handle], 1)
    handles.append(copy[0])

# Move and rename
for handle, pos, name in zip(handles, positions, names):
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectAlias(handle, name)  # 👈 rename root object

sim.startSimulation()



print("Rovers renamed and positioned")