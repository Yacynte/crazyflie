import logging
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


# ─────────────────────────────────────────────────────────────────────────────
# State: wraps the onboard Kalman EKF (kalman.stateZ / kalman.statePZ)
#
# Why not a host-side complementary filter?
#   The barometer spikes hard from propwash (~15 m/s apparent baro_vz),
#   which overwhelms any host-side filter at 20Hz. The onboard EKF runs
#   at 1kHz on the STM32 and fuses baro with a much lower gain, so it
#   handles propwash without the ±1.5m z spikes we saw.
#
# We still need a z_offset: kalman.stateZ is not zeroed at boot —
# it drifts up from whatever the EKF initializes to. We capture the
# ground value on first callback and subtract it everywhere.
# ─────────────────────────────────────────────────────────────────────────────

class KalmanState:
    def __init__(self):
        self.lock        = threading.Lock()
        self.initialized = False
        self._z_offset   = None
        self.z           = 0.0   # relative altitude, m
        self.vz          = 0.0   # vertical velocity,  m/s

    def callback(self, timestamp, data, _logconf):
        with self.lock:
            z_raw = data['kalman.stateZ']
            if self._z_offset is None:
                self._z_offset   = z_raw
                self.initialized = True
                print(f"[kalman] ground z = {z_raw:.3f} m (offset captured)")
            self.z  = z_raw - self._z_offset
            self.vz = data['kalman.statePZ']

    def state(self):
        with self.lock:
            return self.z, self.vz


# ─────────────────────────────────────────────────────────────────────────────
# Controller: PD on altitude error → vz command
#
# Output is a velocity setpoint (m/s), not a thrust value.
# The onboard attitude + thrust controller then tracks that velocity.
#
# Tuning guide:
#   kp  — how aggressively to chase altitude error.
#          Too high → oscillation. Start at 0.15, raise slowly.
#   kd  — damps approach velocity. Prevents overshoot.
#          Match to kp: roughly kd ≈ 0.5–1.0 × kp.
#   vz_max — hard clamp on output. 0.3 m/s is gentle; raise once stable.
# ─────────────────────────────────────────────────────────────────────────────

class AltitudeController:
    def __init__(self, kp=0.2, kd=0.15, vz_max=0.3):
        self.kp     = kp
        self.kd     = kd
        self.vz_max = vz_max

    def compute(self, z_target, z, vz):
        cmd = self.kp * (z_target - z) - self.kd * vz
        return float(np.clip(cmd, -self.vz_max, self.vz_max))


# ─────────────────────────────────────────────────────────────────────────────
# Logging: two LogConfigs
#   - Kalman (100Hz): the state we actually control from
#   - Baro   (50Hz):  kept for debugging / offline analysis only
#                     (not fed into the controller)
#
# LogConfig payload cap: 26 bytes = 6 × float32.
# We stay well under with both configs.
# ─────────────────────────────────────────────────────────────────────────────

def start_logging(scf, kalman):
    kalman_conf = LogConfig(name='Kalman', period_in_ms=10)   # 100 Hz
    kalman_conf.add_variable('kalman.stateZ',  'float')
    kalman_conf.add_variable('kalman.statePZ', 'float')

    baro_conf = LogConfig(name='Baro', period_in_ms=20)       # 50 Hz — debug only
    baro_conf.add_variable('baro.asl', 'float')

    for conf in [kalman_conf, baro_conf]:
        scf.cf.log.add_config(conf)

    kalman_conf.data_received_cb.add_callback(kalman.callback)

    # Baro: just print for now so you can see propwash spikes vs kalman
    def baro_cb(ts, data, _):
        pass   # swap in a CSV logger here if you want offline analysis
    baro_conf.data_received_cb.add_callback(baro_cb)

    kalman_conf.start()
    baro_conf.start()

    return kalman_conf, baro_conf


# ─────────────────────────────────────────────────────────────────────────────
# Flight: three phases
#
#   RAMP  — z_target climbs at TARGET_RATE m/s toward z_final.
#            Avoids saturating the controller with a step demand.
#
#   HOVER — z_target fixed at z_final for HOVER_TIME seconds.
#
#   LAND  — z_target descends at TARGET_RATE m/s back to ground.
#            Loop exits when target reaches ground + 5cm margin,
#            then send_stop_setpoint() cuts the motors cleanly.
# ─────────────────────────────────────────────────────────────────────────────

TARGET_HEIGHT = 0.5   # m above ground
TARGET_RATE   = 0.1   # m/s for ramp and land phases
HOVER_TIME    = 6.0   # seconds
CONTROL_HZ    = 20    # host-side loop rate
DT            = 1.0 / CONTROL_HZ


def fly(scf, kalman):
    cf  = scf.cf
    ctl = AltitudeController(kp=0.2, kd=0.15, vz_max=0.3)

    # Mandatory unlock: firmware ignores velocity setpoints until it sees
    # at least one legacy setpoint (even all-zeros).
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    # Wait for the Kalman offset to be captured (first callback)
    print("Waiting for Kalman state...")
    while not kalman.initialized:
        time.sleep(0.02)
    time.sleep(0.3)   # let the EKF settle before we start moving
    print("Kalman ready — starting flight.")

    z_ground = kalman.state()[0]          # should be ≈ 0.0 after offset
    z_final  = z_ground + TARGET_HEIGHT
    z_target = z_ground

    phase    = 'ramp'
    t_hover  = None

    try:
        while True:
            z, vz = kalman.state()

            # ── phase transitions ──────────────────────────────────────
            if phase == 'ramp':
                z_target = min(z_target + TARGET_RATE * DT, z_final)
                if z_target >= z_final:
                    phase   = 'hover'
                    t_hover = time.time()
                    print(f"Hovering at {z_final:.2f} m")

            elif phase == 'hover':
                z_target = z_final
                if time.time() - t_hover >= HOVER_TIME:
                    phase = 'land'
                    print("Landing...")

            elif phase == 'land':
                z_target = max(z_target - TARGET_RATE * DT, z_ground)
                if z_target <= z_ground + 0.05:
                    break   # close enough to ground — stop motors

            # ── control ───────────────────────────────────────────────
            vz_cmd = ctl.compute(z_target, z, vz)
            cf.commander.send_velocity_world_setpoint(0, 0, vz_cmd, 0)

            print(f"[{phase:5s}] tgt={z_target:+.3f}  z={z:+.3f}  vz={vz:+.3f}  cmd={vz_cmd:+.3f}")
            time.sleep(DT)

    finally:
        # Always reached — even on exception or KeyboardInterrupt.
        # Clears the setpoint so the firmware doesn't hold the last velocity.
        cf.commander.send_stop_setpoint()
        print("Motors stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    kalman = KalmanState()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        logs = start_logging(scf, kalman)
        time.sleep(0.3)

        try:
            fly(scf, kalman)
        finally:
            for log in logs:
                log.stop()