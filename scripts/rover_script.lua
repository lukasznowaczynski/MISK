-- Paste this as a Non-threaded Child Script inside the rover model in CoppeliaSim.
-- Menu: [right-click rover] > Add > Script > Non-threaded child script

function sysCall_init()
    local script_obj = sim.getObject('.')
    self_handle = sim.getObjectParent(script_obj)

    print('Script alias: ' .. sim.getObjectAlias(script_obj, 1))
    print('Parent alias: ' .. sim.getObjectAlias(self_handle, 1))

    rover_name = sim.getObjectAlias(self_handle, 0)
    print('Rover name  : ' .. rover_name)

    -- Print every object in the rover tree with its type
    -- object types: 0=shape, 1=joint, 2=graph, 3=camera, 4=dummy, 7=forceSensor, 10=script
    local all = sim.getObjectsInTree(self_handle, sim.handle_all, 0)
    print('=== ALL objects in rover tree ===')
    for _, obj in ipairs(all) do
        print(sim.getObjectAlias(obj, 1) .. '  [type=' .. sim.getObjectType(obj) .. ']')
    end
    print('=== End ===')
end

function sysCall_actuation()
    -- Guard: joints not yet found
    if not fl then return end

    local left_vel  = sim.getFloatSignal(rover_name .. '_lv') or 0
    local right_vel = sim.getFloatSignal(rover_name .. '_rv') or 0

    sim.setJointTargetVelocity(fl, left_vel)
    sim.setJointTargetVelocity(rl, left_vel)
    sim.setJointTargetVelocity(fr, right_vel)
    sim.setJointTargetVelocity(rr, right_vel)
end

function sysCall_cleanup()
    -- Stop wheels on script removal
    for _, j in ipairs({fl, fr, rl, rr}) do
        sim.setJointTargetVelocity(j, 0)
    end
end
