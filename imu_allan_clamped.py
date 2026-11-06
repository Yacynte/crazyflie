"""
IMU noise with the motors running: the drone clamped down, motors at about
hover thrust. Same output format as imu_allan.py, so the same analysis and
KFConfig mapping apply.

    python imu_allan_clamped.py record clamped.npz --thrust 34000 --minutes 4
    python imu_allan_clamped.py analyze clamped.npz [--static imu_noise.json] [--plot]
                                                    -> imu_noise_clamped.json

Why: the motors-off Allan fit (imu_allan.py) gave acc_vrw 0.0016, the flight
replay (kf_replay.py) needed ~100x more. Propeller vibration, aliased into the
100 Hz log, is the obvious suspect; this measures it without flying.

Motor drive (--mode):
    power     (default) the firmware's motorPowerSet params: the same PWM on
              all four motors, no controller in the loop. A clamped drone
              can't level itself, so an attitude controller would wind up and
              drive the motors unevenly.
    setpoint  roll/pitch 0 attitude setpoints at --thrust, the controller in
              the loop (closer to flight, but integrators wind up against the
              clamp). Use it if the motors don't spin in power mode — some
              firmware gates motorPowerSet behind arming / the supervisor.

Only the steady hold is recorded (not the ramps). It stops early if the
battery sags below --min-vbat: a Crazyflie at hover thrust empties a battery
in ~5-7 min, so long logs aren't possible — expect no bias random walk
estimate, but the vibration (white / short-correlation noise) is what this is
for. Also logged: roll/pitch, whose wander under vibration is the tilt
leakage that dominated the flight replay.

Caveat: a clamp changes the airframe's vibration modes (stiffer, damped), so
this is closer to flight than motors-off, not identical to it. Compare the
result with the kf_replay.py --fit numbers.

SAFETY: the drone must be clamped so it cannot lift or flip, propellers clear
of everything (and of you). Ctrl-C stops the motors immediately.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from imu_allan import G, analyze, plot, print_report, to_kf_params

RAMP_TIME = 3.0     # s, up and down
SETPOINT_HZ = 50    # setpoint mode: the commander watchdog needs a steady stream


class MotorDrive:
    """Spins all four motors at one level, by motorPowerSet or attitude setpoints."""

    def __init__(self, cf, mode):
        self.cf, self.mode = cf, mode
        self.level = 0
        if mode == 'power' and cf.param.toc.get_element_by_complete_name(
                'motorPowerSet.enable') is None:
            raise RuntimeError("firmware has no motorPowerSet params — use --mode setpoint")

    def start(self):
        self.cf.supervisor.send_arming_request(True)
        time.sleep(0.5)
        if self.mode == 'setpoint':
            self.cf.commander.send_setpoint(0, 0, 0, 0)     # releases the thrust lock
        else:
            self.set(0)
            self.cf.param.set_value('motorPowerSet.enable', '1')

    def set(self, level):
        self.level = int(level)
        if self.mode == 'power':
            for m in ('m1', 'm2', 'm3', 'm4'):
                self.cf.param.set_value(f'motorPowerSet.{m}', str(self.level))

    def keepalive(self):
        if self.mode == 'setpoint':
            self.cf.commander.send_setpoint(0, 0, 0, self.level)

    def stop(self):
        self.level = 0
        try:
            if self.mode == 'power':
                self.set(0)
                self.cf.param.set_value('motorPowerSet.enable', '0')
            else:
                self.cf.commander.send_stop_setpoint()
        finally:
            self.cf.supervisor.send_arming_request(False)


def record_clamped(path, minutes, thrust, mode, uri, min_vbat):
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    print(__doc__.split("SAFETY:")[1].strip())
    if input("Clamped, propellers clear? Type 'clamped' to spin the motors: ").strip() != 'clamped':
        print("not started")
        return

    cflib.crtp.init_drivers()
    state = {"hold": False, "vbat": None}
    acc_rows, att_rows, vbat_rows = [], [], []

    def acc_cb(ts, d, _):
        if state["hold"]:
            acc_rows.append((ts / 1000.0, d['acc.x'], d['acc.y'], d['acc.z']))

    def att_cb(ts, d, _):
        state["vbat"] = d['pm.vbat']
        if state["hold"]:
            att_rows.append((ts / 1000.0, d['stabilizer.roll'], d['stabilizer.pitch']))
            vbat_rows.append((ts / 1000.0, d['pm.vbat']))

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        confs = []
        for name, variables, cb in (('Clamp_acc', ('acc.x', 'acc.y', 'acc.z'), acc_cb),
                                    ('Clamp_att', ('stabilizer.roll', 'stabilizer.pitch',
                                                   'pm.vbat'), att_cb)):
            conf = LogConfig(name=name, period_in_ms=10)
            for v in variables:
                conf.add_variable(v, 'float')
            cf.log.add_config(conf)
            conf.data_received_cb.add_callback(cb)
            conf.start()
            confs.append(conf)
        time.sleep(0.5)
        print(f"[clamped] battery {state['vbat']:.2f} V")

        drive = MotorDrive(cf, mode)
        reason = "done"
        try:
            drive.start()
            print(f"[clamped] ramping to {thrust} ({mode} mode)")
            t0 = time.time()
            while time.time() - t0 < RAMP_TIME:
                drive.set(thrust * (time.time() - t0) / RAMP_TIME)
                drive.keepalive()
                time.sleep(1.0 / SETPOINT_HZ)
            drive.set(thrust)
            time.sleep(1.0)                          # let the vibration settle
            state["hold"] = True

            t_end = time.time() + minutes * 60
            last_print = 0.0
            while time.time() < t_end:
                drive.keepalive()
                if state["vbat"] is not None and state["vbat"] < min_vbat:
                    reason = f"battery {state['vbat']:.2f} V < {min_vbat} V"
                    break
                if time.time() - last_print > 1.0:
                    last_print = time.time()
                    recent = np.array([r[1:] for r in acc_rows[-100:]]) * G
                    vib = recent.std(axis=0).max() if len(recent) > 10 else float('nan')
                    print(f"\r[clamped] {(t_end - time.time()) / 60:4.1f} min left, "
                          f"{len(acc_rows)} samples, vibration {vib:.2f} m/s^2, "
                          f"battery {state['vbat']:.2f} V   ", end='', flush=True)
                time.sleep(1.0 / SETPOINT_HZ)
        except KeyboardInterrupt:
            reason = "Ctrl-C"
            drive.stop()                             # immediately, no ramp
        finally:
            state["hold"] = False
            if drive.level > 0:
                start = drive.level
                t0 = time.time()
                while time.time() - t0 < RAMP_TIME:
                    drive.set(start * (1 - (time.time() - t0) / RAMP_TIME))
                    drive.keepalive()
                    time.sleep(1.0 / SETPOINT_HZ)
            drive.stop()
            for conf in confs:
                conf.stop()
        print(f"\n[clamped] stopped: {reason}")

    if len(acc_rows) < 1000:
        print(f"[clamped] only {len(acc_rows)} samples — not saved")
        return
    acc = np.array(acc_rows)
    att = np.array(att_rows)
    vb = np.array(vbat_rows)
    np.savez(path, t_acc=acc[:, 0], acc=acc[:, 1:] * G,
             t_att=att[:, 0], att=np.radians(att[:, 1:]),
             t_vbat=vb[:, 0], vbat=vb[:, 1], thrust=thrust, mode=mode)
    print(f"[clamped] saved {len(acc)} samples ({np.ptp(acc[:, 0]) / 60:.1f} min) to {path}, "
          f"battery {vb[0, 1]:.2f} -> {vb[-1, 1]:.2f} V")


def analyze_clamped(path, out, static=None, show=False):
    d = np.load(path)
    params, curves = analyze(d['acc'], d['t_acc'], d['att'], d['t_att'])
    kf = to_kf_params(params)
    if params["duration_s"] < 600 and kf.pop("bias_rw", None) is not None:
        # a battery-length log can't resolve a random walk (synthetic 4 min: 5x off);
        # leaving it out keeps the KFConfig / motors-off value
        print("[clamped] log < 10 min: bias_rw not estimated, keep the motors-off value")
    print(f"\nclamped, thrust {int(d['thrust'])} ({d['mode']} mode)")
    print_report(params, kf)
    if static:
        with open(static) as fh:
            ref = json.load(fh).get("kf", {})
        print(f"\n{'':11s} {'motors off':>11s} {'clamped':>11s} {'ratio':>7s}")
        for name, v in kf.items():
            if name in ref and ref[name]:
                print(f"{name:11s} {ref[name]:11.4g} {v:11.4g} {v / ref[name]:7.1f}")
    with open(out, 'w') as fh:
        json.dump({"kf": kf, "allan": params,
                   "source": f"imu_allan_clamped.py, thrust {int(d['thrust'])}"}, fh, indent=2)
    print(f"written to {out} — load with KFConfig.from_json('{out}')")
    if show:
        plot(curves, params)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['record', 'analyze'])
    ap.add_argument('file')
    ap.add_argument('--thrust', type=int, default=34000,
                    help='per-motor PWM / thrust; flight3 hovered at ~34400')
    ap.add_argument('--minutes', type=float, default=4.0)
    ap.add_argument('--mode', choices=['power', 'setpoint'], default='power')
    ap.add_argument('--min-vbat', type=float, default=3.4, help='V, stop below this')
    ap.add_argument('--static', help='motors-off imu_noise.json to compare with')
    ap.add_argument('--out', default='imu_noise_clamped.json')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    if args.cmd == 'record':
        if not 10001 <= args.thrust <= 60000:
            sys.exit("--thrust must be in 10001..60000")
        from cflib.utils import uri_helper
        record_clamped(args.file, args.minutes, args.thrust, args.mode,
                       uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702'),
                       args.min_vbat)
    else:
        analyze_clamped(args.file, args.out, args.static, args.plot)
