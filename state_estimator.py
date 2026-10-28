import logging
import sys
import time
import threading
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702')
logging.basicConfig(level=logging.ERROR)


class ZEstimator:
    G     = 9.81
    ALPHA = 0.97

    def __init__(self):
        self.lock = threading.Lock()

        # Two-phase init: wait for baro first, THEN start integrating
        self._baro_ready = False
        self._imu_ready  = False
        self.initialized = False

        self.acc   = np.zeros(3)
        self.roll  = 0.0
        self.pitch = 0.0
        self.yaw   = 0.0

        self.baro_asl    = None   # raw ASL reading, m
        self.z_offset    = None   # ASL at ground — subtracted to give relative z
        self.baro_z      = 0.0   # relative baro altitude
        self.baro_z_prev = 0.0

        self.z      = 0.0
        self.vz     = 0.0
        self.t_prev = None

    # ── Phase 1: baro callback fires first ──────────────────────────
    def baro_callback(self, timestamp, data, _logconf):
        with self.lock:
            self.baro_asl = data['baro.asl']
            if not self._baro_ready:
                # Capture ground-level ASL the first time we hear from baro
                self.z_offset    = self.baro_asl
                self.baro_z      = 0.0
                self.baro_z_prev = 0.0
                self.z           = 0.0
                self._baro_ready = True
                print(f"[estimator] baro ready — ground ASL = {self.z_offset:.2f} m")
            else:
                self.baro_z = self.baro_asl - self.z_offset   # relative!

    # ── Phase 2: IMU callback — only integrates once baro is ready ──
    def imu_callback(self, timestamp, data, _logconf):
        with self.lock:
            if not self._baro_ready:
                return     # don't touch anything until baro baseline is set

            t = timestamp / 1000.0

            self.acc   = np.array([data['acc.x'],
                                   data['acc.y'],
                                   data['acc.z']]) * self.G
            self.roll  = np.radians(data['stabilizer.roll'])
            self.pitch = np.radians(data['stabilizer.pitch'])
            self.yaw   = np.radians(data['stabilizer.yaw'])

            if not self._imu_ready:
                self.t_prev     = t
                self._imu_ready = True
                self.initialized = True
                return

            dt = t - self.t_prev
            if dt <= 0 or dt > 0.5:
                self.t_prev = t
                return

            az_world = (self._R() @ self.acc)[2] - self.G
            # baro_vz  = (self.baro_z - self.baro_z_prev) / dt
            raw_baro_vz = (self.baro_z - self.baro_z_prev) / dt
            baro_vz = float(np.clip(raw_baro_vz, -0.5, 0.5))  # reject propwash spikes

            vz_pred   = self.vz + az_world * dt
            self.vz   = self.ALPHA * vz_pred + (1 - self.ALPHA) * baro_vz
            self.z    = self.ALPHA * (self.z + self.vz * dt) + (1 - self.ALPHA) * self.baro_z

            self.baro_z_prev = self.baro_z
            self.t_prev      = t

    def _R(self):
        r, p, y = self.roll, self.pitch, self.yaw
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,     cp*sr,             cp*cr            ],
        ])

    def state(self):
        with self.lock:
            return self.z, self.vz


class AltitudeController:
    """
    PD on position error → vz command.
    kp low: the drone is slow to reach target but won't oscillate.
    kd damps the approach — keeps it from overshooting.
    Raise kp only after you confirm the estimator is stable.
    """
    def __init__(self, kp=0.15, kd=0.1, vz_max=0.3):
        self.kp     = kp
        self.kd     = kd
        self.vz_max = vz_max

    def compute(self, z_target, z, vz):
        cmd = self.kp * (z_target - z) - self.kd * vz
        return float(np.clip(cmd, -self.vz_max, self.vz_max))


def start_logging(scf, estimator):
    imu_conf = LogConfig(name='IMU', period_in_ms=10)
    for v in ['acc.x', 'acc.y', 'acc.z',
              'stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw']:
        imu_conf.add_variable(v, 'float')

    baro_conf = LogConfig(name='Baro', period_in_ms=20)
    baro_conf.add_variable('baro.asl', 'float')

    # Also log the onboard Kalman estimate — good sanity check
    kalman_conf = LogConfig(name='Kalman', period_in_ms=20)
    kalman_conf.add_variable('kalman.stateZ',  'float')
    kalman_conf.add_variable('kalman.statePZ', 'float')

    for conf in [imu_conf, baro_conf, kalman_conf]:
        scf.cf.log.add_config(conf)

    imu_conf.data_received_cb.add_callback(estimator.imu_callback)
    baro_conf.data_received_cb.add_callback(estimator.baro_callback)

    # Print Kalman alongside ours so you can compare in real time
    def kalman_cb(ts, data, _):
        pass  # hook up your logger / CSV here if you want
    kalman_conf.data_received_cb.add_callback(kalman_cb)

    imu_conf.start()
    baro_conf.start()
    kalman_conf.start()
    return imu_conf, baro_conf, kalman_conf


def fly(scf, estimator):
    cf  = scf.cf
    ctl = AltitudeController(kp=0.15, kd=0.1, vz_max=0.3)
    dt  = 0.05  # 20 Hz

    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    # Wait for estimator to be ready (baro baseline + first IMU)
    print("Waiting for estimator init...")
    while not estimator.initialized:
        time.sleep(0.05)
    print("Estimator ready.")
    time.sleep(0.5)   # let it settle before moving

    z0 = estimator.state()[0]
    z_final = z0 + 0.5  # 50 cm above wherever we are right now

    # Ramp the target up at 0.1 m/s — don't step it to 0.5 immediately
    TARGET_RATE = 0.1   # m/s upward ramp
    z_target    = z0
    t0          = time.time()
    RAMP_TIME   = z_final / TARGET_RATE   # time to reach z_final
    HOVER_TIME  = 5.0

    try:
        phase = 'ramp'
        while True:
            elapsed = time.time() - t0
            z, vz = estimator.state()

            if phase == 'ramp':
                z_target = z0 + TARGET_RATE * elapsed
                if z_target >= z_final:
                    z_target = z_final
                    phase = 'hover'
                    print(f"Hovering at {z_final:.2f} m relative")
                    t_hover = time.time()

            elif phase == 'hover':
                z_target = z_final
                if time.time() - t_hover > HOVER_TIME:
                    phase = 'land'
                    print("Landing...")

            elif phase == 'land':
                z_target = max(z_target - TARGET_RATE * dt, z0)
                if z_target <= z0 + 0.05:
                    break

            vz_cmd = ctl.compute(z_target, z, vz)
            cf.commander.send_velocity_world_setpoint(0, 0, vz_cmd, 0)

            print(f"[{phase}] z_tgt={z_target:+.3f} z={z:+.3f} vz={vz:+.3f} cmd={vz_cmd:+.3f}")
            time.sleep(dt)

    finally:
        cf.commander.send_stop_setpoint()


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    estimator = ZEstimator()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        logs = start_logging(scf, estimator)
        time.sleep(0.3)

        try:
            fly(scf, estimator)
        finally:
            for l in logs:
                l.stop()