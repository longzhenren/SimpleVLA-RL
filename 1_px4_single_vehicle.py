#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with a single vehicle, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

import sys, os
# 把 Pegasus 包所在的上级目录插到最前面
sys.path.insert(0, os.path.expanduser("~/PegasusSimulator/extensions/pegasus.simulator/"))


# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

# Auxiliary scipy and numpy modules
import os.path
from scipy.spatial.transform import Rotation

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()
        # generate a timestamp for saving images and sensor data
        import datetime
        self.timestamp = datetime.datetime.now()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse"])

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        config_multirotor.sensors = [
            Barometer(config={"update_rate": 50.0}),
            IMU(config={"update_rate": 100.0}),
            Magnetometer(config={"update_rate": 50.0}),
            GPS(config={"update_rate": 10.0})
        ]
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        self._PX4Backend = PX4MavlinkBackend(mavlink_config)
        config_multirotor.backends = [self._PX4Backend]

        config_multirotor.graphical_sensors = [MonocularCamera("front_camera", config={"update_rate": 60.0,})]
        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
            #从pgSim API获取传感器数据，每100次保存一次
            if int(self.timeline.get_current_time() * 60) % 100 == 0:
                carb.log_info("Saving sensor data at time: {}".format(self.timeline.get_current_time()))
                import json
                # sensors = self.pg.get_vehicle("/World/quadrotor")._sensors
                data = self._PX4Backend._sensor_data
                sensors_data = {}
                sensors_data["imu"] = {"xacc":data.xacc, "yacc":data.yacc, "zacc":data.zacc,"xgyro":data.xgyro, "ygyro":data.ygyro, "zgyro":data.zgyro}
                sensors_data["gps"] = {"fix_type": data.fix_type,"latitude_deg": data.latitude_deg,"longitude_deg": data.longitude_deg,"altitude": data.altitude,"eph": data.eph,"epv": data.epv,"velocity": data.velocity,"velocity_north": data.velocity_north,"velocity_east": data.velocity_east,"velocity_down": data.velocity_down,"cog": data.cog,"sim_lat": data.sim_lat,"sim_lon": data.sim_lon,"sim_alt": data.sim_alt}
                sensors_data["barometer"] = {"abs_pressure": data.abs_pressure, "temperature": data.temperature, "pressure_alt": data.pressure_alt}
                sensors_data["magnetometer"] = {"xmag": data.xmag, "ymag": data.ymag, "zmag": data.zmag}
                sensors_data["sim"] = {"sim_attitude": data.sim_attitude, "sim_angular_vel": data.sim_angular_vel, "sim_acceleration": data.sim_acceleration, "sim_velocity_inertial": data.sim_velocity_inertial,"sim_ind_airspeed": data.sim_ind_airspeed, "sim_true_airspeed": data.sim_true_airspeed}
                # write to json file
                os.makedirs("sensor_data/sensors_{}".format(self.timestamp), exist_ok=True)
                with open("sensor_data/sensors_{}/sensors_data_{}.json".format(self.timestamp,int(self.timeline.get_current_time())), "w") as f:
                    json.dump(sensors_data, f)
            # save the image to a file with filename indexed by the current timeline timestamp every 60 iterations
            if int(self.timeline.get_current_time() * 60) % 60 == 0:
                from PIL import Image
                carb.log_info("Saving camera image at time: {}".format(self.timeline.get_current_time()))
                os.makedirs("sensor_data/camera_{}".format(self.timestamp), exist_ok=True)
                img = Image.fromarray(self.pg.get_vehicle("/World/quadrotor")._graphical_sensors[0]._camera.get_rgba()[:, :, :3])
                img.save("sensor_data/camera_{}/camera_image_{}.png".format(self.timestamp,int(self.timeline.get_current_time())))
                
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
