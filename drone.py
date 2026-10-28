import logging
import time
import threading
from threading import Event, Barrier

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper


class AllDrones:
    def __init__(self):
        self.drones = []
        self.barrier = None  # Initialize the barrier to None

    def add_drone(self, drone):
        self.drones.append(drone)
        self.barrier = Barrier(len(self.drones))

class Drone(AllDrones):
    def __init__(self, uri, drone_name):
        self.uri = uri
        self.drone_name = drone_name
        self.deck_attached_event = Event()
        self.setup_drone()  # Call setup_drone during initialization

    def param_deck_flow(self, _, value_str):
        value = int(value_str)

        if value:
            print(f'{self.drone_name}: Flow deck attached')
            self.deck_attached_event.set()
        else:
            print(f'{self.drone_name}: Flow deck NOT attached')

    def setup_drone(self):
        try:
            print(f'{self.drone_name}: connecting...')

            with SyncCrazyflie(
                self.uri,
                cf=Crazyflie(rw_cache='./cache')
            ) as scf:

                print(f'{self.drone_name}: connected')

                # Register Flow deck callback
                scf.cf.param.add_update_callback(
                    group='deck',
                    name='bcFlow2',
                    cb=self.param_deck_flow
                )
                # Give parameter system time to report
                time.sleep(1)
    
                # Wait for THIS drone's Flow deck
                if not self.deck_attached_event.wait(timeout=5):
                    raise RuntimeError(
                        f'{self.drone_name}: No Flow deck detected!'
                    )
    
                print(f'{self.drone_name}: ready')
        except Exception as e:
            print(f'{self.drone_name}: Error during setup: {e}')

    def get_drone_location(self):
        # Placeholder for getting the drone's location
        # In a real implementation, this would return the actual position
        return (0, 0, 0)  # Example: (x, y, z) coordinates