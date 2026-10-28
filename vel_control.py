import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.positioning.motion_commander import MotionCommander

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702')
DEFAULT_HEIGHT = 0.5

logging.basicConfig(level=logging.ERROR)


def send_vel(cf, vx, vy, vz, yaw_rate, duration, dt=0.05):
    """Send a constant velocity setpoint for `duration` seconds."""
    steps = int(duration / dt)
    for _ in range(steps):
        cf.commander.send_velocity_world_setpoint(vx, vy, vz, yaw_rate)
        time.sleep(dt)

def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        mc.up(0.3)
        time.sleep(3)


def fly_velocity(scf):
    cf = scf.cf

    # Unlock the commander — mandatory zero setpoint before any motion
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    try:
        send_vel(cf, 0, 0,  0.3, 0, duration=2.0)   # climb
        send_vel(cf, 0, 0,  0.0, 0, duration=3.0)   # hover
        send_vel(cf, 0, 0, -0.3, 0, duration=2.0)   # descend
    finally:
        cf.commander.send_stop_setpoint()


if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        # fly_velocity(scf)
        take_off_simple(scf)