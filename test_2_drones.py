import logging
import time
import threading
from threading import Event, Barrier

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper


URI = uri_helper.uri_from_env(
    default='radio://0/80/2M/E7E7E7E701'
)

URI2 = uri_helper.uri_from_env(
    default='radio://0/80/2M/E7E7E7E702'
)

DEFAULT_HEIGHT = 0.5

logging.basicConfig(level=logging.ERROR)


# Two drones -> barrier needs 2 participants
takeoff_barrier = Barrier(2)


def setup_drone(uri, drone_name):

    # Each drone gets its own event
    deck_attached_event = Event()

    def param_deck_flow(_, value_str):
        value = int(value_str)

        if value:
            print(f'{drone_name}: Flow deck attached')
            deck_attached_event.set()
        else:
            print(f'{drone_name}: Flow deck NOT attached')

    try:
        print(f'{drone_name}: connecting...')

        with SyncCrazyflie(
            uri,
            cf=Crazyflie(rw_cache='./cache')
        ) as scf:

            print(f'{drone_name}: connected')

            # Register Flow deck callback
            scf.cf.param.add_update_callback(
                group='deck',
                name='bcFlow2',
                cb=param_deck_flow
            )

            # Give parameter system time to report
            time.sleep(1)

            # Wait for THIS drone's Flow deck
            if not deck_attached_event.wait(timeout=5):
                raise RuntimeError(
                    f'{drone_name}: No Flow deck detected!'
                )

            print(f'{drone_name}: ready')

            # Arm this drone
            print(f'{drone_name}: arming...')
            scf.cf.supervisor.send_arming_request(True)

            time.sleep(1)

            print(f'{drone_name}: armed')

            # -------------------------------------------------
            # WAIT HERE
            #
            # Drone 1 waits for Drone 2
            # Drone 2 waits for Drone 1
            # -------------------------------------------------

            print(f'{drone_name}: waiting for other drone...')

            takeoff_barrier.wait()

            # -------------------------------------------------
            # BOTH DRONES ARE RELEASED HERE
            # -------------------------------------------------

            print(f'{drone_name}: TAKE OFF!')

            with MotionCommander(
                scf,
                default_height=DEFAULT_HEIGHT
            ) as mc:

                # Hover for 3 seconds
                time.sleep(3)

            # Leaving MotionCommander causes the drone to land

            print(f'{drone_name}: landed')

    except threading.BrokenBarrierError:
        print(
            f'{drone_name}: barrier broken - '
            'the other drone may have failed'
        )

    except Exception as e:
        print(f'{drone_name}: ERROR: {e}')


if __name__ == '__main__':

    cflib.crtp.init_drivers()

    thread1 = threading.Thread(
        target=setup_drone,
        args=(URI, 'Drone 1')
    )

    thread2 = threading.Thread(
        target=setup_drone,
        args=(URI2, 'Drone 2')
    )

    # Start both threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    print('Both drones have finished.')
