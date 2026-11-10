"""
Host-side 3D inertial Kalman filter for a Crazyflie, IMU only.

    InertialKF     pure numpy filter, no cflib — testable offline (--sim)
    CrazyflieKF    cflib wiring: log callbacks, calibration, auto zero-velocity

With no aiding sensor nothing corrects the estimate in flight: the filter
propagates the IMU and, as importantly, the uncertainty of every error
source. The accuracy you get is set by how well those errors are modelled
and how well they were calibrated on the ground — so the model is the point.

State (14):

    x = [p(3), v(3), b(3), n(3), e(2)]

    p   position, m (frame: where calibration happened, x along the nose at boot)
    v   velocity, m/s
    b   accelerometer bias, body, m/s^2          random walk        bias_rw
    n   correlated accel noise, body, m/s^2      Gauss-Markov       gm_sigma, gm_tau
    e   horizontal accel error from attitude     Gauss-Markov       tilt_sigma, tilt_tau
        error, world, m/s^2  (gravity leaking through a tilt error: g * dtheta)

plus white accelerometer noise acc_vrw, which isn't a state (it's gone by the
next sample) but drives the growth of v and p uncertainty.

    a_world = R (a_meas - b - n) - g e_z - [e, 0]
    p'      = p + v dt + dt^2/2 a_world
    v'      = v + dt a_world
    b'      = b + w_b                      E[w_b^2] = bias_rw^2 dt
    n'      = exp(-dt/gm_tau) n + w_n      steady-state std gm_sigma
    e'      = exp(-dt/tilt_tau) e + w_e    steady-state std tilt_sigma

R comes from the onboard attitude and is treated as a known input, so the
filter is linear (time-varying KF, not an EKF). The noise parameters come
from an Allan-variance fit of your own drone: see imu_allan.py, then
KFConfig.from_json('imu_noise.json').

Corrections (all on the ground):
    zero velocity     automatic while motors are off and the drone is still —
                      before takeoff this refines b, after landing it stops
                      the drift
    floor height      on landing (motors stop -> still), z = the takeoff floor,
                      after a velocity "shock" that discards the impact
    ground lock       zero velocity + floor height while spinning up on the
                      ground, until liftoff
    fix_position(p)   optional: the drone sits at a known spot (start pad,
                      landing pad)

Drift grows ~ t^2 in flight. uncertainty() is the filter's honest answer to
"how wrong could I be right now".
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, fields, replace

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

import numpy as np

G = 9.81
GRAVITY = np.array([0.0, 0.0, G])

# stabilizer.pitch is logged in the firmware's legacy convention (sign
# inverted w.r.t. right-handed ZYX). Check by sitting the drone on a ~15 deg
# wedge, nose up and nose down: the calibration print "residual" should stay
# ~0 for both. With the wrong sign it reads ~0.6 m/s^2.
PITCH_SIGN = -1.0

P, V, B, N, E = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 14)
NX = 14


@dataclass
class KFConfig:
    # accelerometer error model, body frame — replace with imu_allan.py output
    acc_vrw: float = 0.04           # m/s/sqrt(s), white noise (velocity random walk)
    gm_sigma: float = 0.05          # m/s^2, std of the correlated noise n
    gm_tau: float = 10.0            # s, correlation time of n
    bias_rw: float = 0.002          # m/s^2/sqrt(s), bias random walk

    # attitude error, as world-frame horizontal acceleration (g * tilt error)
    tilt_sigma: float = 0.085       # m/s^2  (~0.5 deg)
    tilt_tau: float = 60.0          # s

    # rotor drag, IMU-only velocity aiding while coasting (0 = off): body x/y
    # accel = -drag_k * body velocity. Fitted by kf_replay.py --drag; applied
    # on drag_every-sample means because the residual is time-correlated
    drag_k: float = 0.0             # 1/s
    drag_sigma: float = 0.6         # m/s^2
    drag_every: int = 10            # IMU samples per update (10 Hz)

    # ground corrections
    sigma_fix: float = 0.01         # m
    sigma_zupt: float = 0.005       # m/s
    gate_chi2: float = 16.27        # 3-dof, 99.9 %

    # initial uncertainty
    p0_pos: float = 0.01 ** 2
    p0_vel: float = 0.01 ** 2
    p0_bias: float = 0.02 ** 2      # what's left after ground calibration

    max_dt: float = 0.1             # s, larger IMU gaps skip the predict

    @classmethod
    def from_json(cls, path, **overrides):
        """Load the 'kf' block written by imu_allan.py; unknown keys are ignored."""
        with open(path) as f:
            d = json.load(f)
        d = d.get("kf", d)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in {**d, **overrides}.items() if k in names})


def rotation(roll, pitch, yaw):
    """Body -> world, ZYX, right-handed, rad."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def calibrate_bias(accs, Rs):
    """Accelerometer bias from a still period: the reading minus R^T g."""
    return np.mean([a - Rm.T @ GRAVITY for a, Rm in zip(accs, Rs)], axis=0)


class InertialKF:
    def __init__(self, cfg: KFConfig | None = None):
        self.cfg = cfg or KFConfig()
        self.x = np.zeros(NX)
        self.P = self._p0()
        self.t = None                    # filter time, s
        self.stats = {k: 0 for k in ("predict", "fix_ok", "fix_rej", "zupt", "dt_skip")}
        self.last_nis = 0.0

    def _p0(self):
        c = self.cfg
        return np.diag([c.p0_pos] * 3 + [c.p0_vel] * 3 + [c.p0_bias] * 3
                       + [c.gm_sigma ** 2] * 3 + [c.tilt_sigma ** 2] * 2)

    def reset(self, p, bias, t):
        self.x = np.zeros(NX)
        self.x[P] = p
        self.x[B] = bias
        self.P = self._p0()
        self.t = t

    # ── predict ─────────────────────────────────────────────────────────────
    def predict(self, acc_body, Rwb, t):
        """Propagate to time t with body accel (m/s^2, incl. gravity) and attitude Rwb."""
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0.0:
            return
        c = self.cfg
        if dt > c.max_dt:                # link hiccup: don't integrate garbage
            self.stats["dt_skip"] += 1
            self.t = t
            self.x[P] += self.x[V] * dt
            self.P[P, P] += np.eye(3) * (np.dot(self.x[V], self.x[V]) * dt ** 2 + c.p0_pos)
            self.P[V, V] += np.eye(3) * c.acc_vrw ** 2 * dt
            return

        dt2 = 0.5 * dt * dt
        phi_n = math.exp(-dt / c.gm_tau)
        phi_e = math.exp(-dt / c.tilt_tau)
        Exy = np.eye(3)[:, :2]           # world horizontal

        a = Rwb @ (acc_body - self.x[B] - self.x[N]) - GRAVITY - Exy @ self.x[E]
        self.x[P] += self.x[V] * dt + dt2 * a
        self.x[V] += a * dt
        self.x[N] *= phi_n
        self.x[E] *= phi_e

        F = np.eye(NX)
        F[P, V] = np.eye(3) * dt
        F[P, B] = F[P, N] = -dt2 * Rwb
        F[V, B] = F[V, N] = -dt * Rwb
        F[P, E] = -dt2 * Exy
        F[V, E] = -dt * Exy
        F[N, N] = np.eye(3) * phi_n
        F[E, E] = np.eye(2) * phi_e

        # white accel noise integrated over dt (continuous-time VRW); isotropic,
        # so the rotation drops out
        Q = np.zeros((NX, NX))
        q = c.acc_vrw ** 2
        Q[P, P] = np.eye(3) * q * dt ** 3 / 3.0
        Q[P, V] = Q[V, P] = np.eye(3) * q * dt ** 2 / 2.0
        Q[V, V] = np.eye(3) * q * dt
        Q[B, B] = np.eye(3) * c.bias_rw ** 2 * dt
        Q[N, N] = np.eye(3) * c.gm_sigma ** 2 * (1.0 - phi_n ** 2)
        Q[E, E] = np.eye(2) * c.tilt_sigma ** 2 * (1.0 - phi_e ** 2)

        self.P = F @ self.P @ F.T + Q
        self.t = t
        self.stats["predict"] += 1

    # ── generic vector update ───────────────────────────────────────────────
    def _update(self, y, H, R, gate=True):
        """y = H x + v, v ~ N(0, R). Returns (accepted, NIS)."""
        nu = y - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        nis = float(nu @ S_inv @ nu)
        if gate and nis > self.cfg.gate_chi2:
            return False, nis

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ nu
        I_KH = np.eye(NX) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return True, nis

    # ── ground corrections ──────────────────────────────────────────────────
    def fix_position(self, p, sigma=None, force=False):
        """The drone is at known spot p (world, m) right now, give or take sigma."""
        sigma = self.cfg.sigma_fix if sigma is None else sigma
        H = np.zeros((3, NX))
        H[:, P] = np.eye(3)
        if force:   # the fix is the truth: forget what we thought about p
            self.P[P, :] = 0.0
            self.P[:, P] = 0.0
            self.P[P, P] = np.eye(3) * 1e2
        ok, nis = self._update(np.asarray(p, float), H, np.eye(3) * sigma ** 2,
                               gate=not force)
        self.last_nis = nis
        self.stats["fix_ok" if ok else "fix_rej"] += 1
        return ok

    def fix_velocity(self, v, sigma, force=False):
        """Velocity from an external reference (onboard estimate while it flies)."""
        H = np.zeros((3, NX))
        H[:, V] = np.eye(3)
        if force:   # the reference is the truth: forget what we thought about v
            self.P[V, :] = 0.0
            self.P[:, V] = 0.0
            self.P[V, V] = np.eye(3) * 1e2
        ok, nis = self._update(np.asarray(v, float), H, np.eye(3) * sigma ** 2,
                               gate=not force)
        self.stats["fix_ok" if ok else "fix_rej"] += 1
        return ok

    def update_drag(self, acc_body, Rwb):
        """Rotor drag: body x/y specific force = -drag_k (R^T v)_xy + b_xy + n_xy."""
        H = np.zeros((2, NX))
        H[:, V] = -self.cfg.drag_k * Rwb.T[:2, :]
        H[0, 6] = H[1, 7] = 1.0          # b_x, b_y
        H[0, 9] = H[1, 10] = 1.0         # n_x, n_y
        self._update(np.asarray(acc_body[:2], float), H,
                     np.eye(2) * self.cfg.drag_sigma ** 2, gate=False)
        self.stats["drag"] = self.stats.get("drag", 0) + 1

    def update_height(self, z, sigma=0.02):
        """The drone sits on a floor of known height z."""
        H = np.zeros((1, NX))
        H[0, 2] = 1.0
        self._update(np.array([z]), H, np.array([[sigma ** 2]]), gate=False)

    def shock(self, sigma_v=1.0):
        """Velocity hit by something the IMU didn't capture (touchdown impact,
        saturation): forget it, and its correlation with everything else, so
        the next zero-velocity update fixes v without dragging p along."""
        self.P[V, :] = 0.0
        self.P[:, V] = 0.0
        self.P[V, V] = np.eye(3) * sigma_v ** 2

    def update_zero_velocity(self):
        H = np.zeros((3, NX))
        H[:, V] = np.eye(3)
        self._update(np.zeros(3), H, np.eye(3) * self.cfg.sigma_zupt ** 2, gate=False)
        self.stats["zupt"] += 1

    def uncertainty(self):
        """1-sigma position uncertainty per axis, m."""
        return np.sqrt(np.diag(self.P)[P])


# ─────────────────────────────────────────────────────────────────────────────
# cflib wiring
#
#   KF_IMU  100 Hz  acc.x/y/z + stabilizer.roll/pitch/yaw    6 floats = 24 B
#   KF_Aux   50 Hz  stabilizer.thrust                         motors off?
#                   stateEstimate.x/y/z                       onboard estimate, reference
#   KF_RefV  50 Hz  stateEstimate.vx/vy/vz                    onboard velocity, reference
#
# Aiding: while `aiding` is True, the onboard estimate (Flow deck — the
# "visual" estimate) is fused as position + velocity measurements. That's
# how the KF learns its bias and tilt error in flight, with the motors
# vibrating, before it is left to fly on the IMU alone.
#
# Startup: the drone sits still at `origin` for CALIB_TIME; the mean
# accelerometer reading minus gravity is the initial bias.
#
# Zero velocity is applied automatically when thrust is 0 AND the
# accelerometer has been quiet for ZUPT_WINDOW samples. Thrust alone isn't
# enough (you may be carrying it); stillness alone isn't either (a smooth
# hover reads ~g too).
#
# Frame: the onboard yaw starts at 0 at boot, so the nose at power-up
# defines +x.
# ─────────────────────────────────────────────────────────────────────────────

# What the host controller flies on (--source):
#   imu     this KF, IMU only — no visual measurement fused while it flies
#   visual  the onboard (Flow deck) estimate, this KF only logged alongside
#   both    this KF with the onboard estimate fused continuously (100 Hz IMU
#           prediction, 50 Hz visual corrections)
SOURCES = ('imu', 'visual', 'both')

# Who is commanding the motors, and on which estimate, recorded on every IMU
# sample (`controller`):
#   none         motors off / stopped
#   visual       the firmware's high-level commander, on the onboard (Flow deck) estimate
#   host         this file's controller / sink / landing, on this KF
#   host_visual  this file's controller, on the onboard estimate (--source visual)
# "Flying on the IMU alone" = controller 'host' AND aiding off AND the
# visual-correction counter not moving. flight_report() checks exactly that.
CONTROLLERS = ('none', 'visual', 'host', 'host_visual')


def host_controller(source):
    """The `controller` tag for this file's controller flying on `source`."""
    return 'host_visual' if source == 'visual' else 'host'
SOURCE_LABEL = {'imu': 'KF (IMU only)', 'visual': 'visual (Flow deck)',
                'both': 'KF (IMU + visual)'}


class CrazyflieKF:
    CALIB_TIME = 1.5        # s on the ground, motionless
    REF_SIGMA_P = 0.03      # m, onboard position as a measurement while aiding
    REF_SIGMA_V = 0.05      # m/s
    REANCHOR_AFTER = 25     # consecutive rejected reference fixes (0.5 s) -> re-anchor
    VIB_WINDOW = 100        # IMU samples (1 s) for the in-flight vibration estimate
    ZUPT_WINDOW = 50        # samples (0.5 s at 100 Hz)
    ZUPT_ACC_STD = 0.1      # m/s^2, per axis, "still"
    ZUPT_EVERY = 5          # apply on every 5th still sample

    def __init__(self, origin=(0.0, 0.0, 0.0), cfg: KFConfig | None = None):
        self.lock = threading.Lock()
        self.kf = InertialKF(replace(cfg or KFConfig()))   # own copy: acc_vrw adapts in flight
        self.base_vrw = self.kf.cfg.acc_vrw
        self.origin = np.asarray(origin, float)
        self.initialized = False
        self.motors_off = True
        self.still = False
        self.yaw = 0.0                  # rad, onboard attitude
        self.az_world = 0.0             # m/s^2, vertical specific force minus g (touchdown)
        self._az_window = deque(maxlen=20)   # 0.2 s, liftoff detection
        self.floor_z = self.origin[2]   # land on the takeoff floor
        self._landed_pending = False
        self.onboard_p = None           # onboard (Flow deck) estimate
        self.onboard_v = None
        self.aiding = False             # fuse the onboard estimate as measurements
        self.controller = 'none'        # who commands the motors (CONTROLLERS), set by the flight code
        self.visual_fixes = 0           # visual measurements actually applied to the KF, ever
        self.imu_only = False           # lock_imu_only(): no visual measurement can reach the KF
        self._rej_streak = 0
        self.reanchors = 0
        self.thrust = 0.0               # last logged stabilizer.thrust
        self._acc_window = deque(maxlen=self.ZUPT_WINDOW)
        self._n_imu = 0
        self._calib = {"acc": [], "R": [], "t0": None}
        self.history = None          # set to [] to record (t, p, 1-sigma, aiding, controller,
                                     # visual_fixes) per IMU sample
        self.imu_history = None      # set to [] to record raw (t, acc, roll, pitch, yaw)
        self._vib = deque(maxlen=self.VIB_WINDOW)
        self._drag_buf = []
        self.ref_history = None      # set to [] to record (t, onboard p, thrust)

    def imu_callback(self, timestamp, data, _logconf):
        t = timestamp / 1000.0
        acc = np.array([data['acc.x'], data['acc.y'], data['acc.z']]) * G
        Rwb = rotation(math.radians(data['stabilizer.roll']),
                       PITCH_SIGN * math.radians(data['stabilizer.pitch']),
                       math.radians(data['stabilizer.yaw']))
        with self.lock:
            if not self.initialized:
                self._calibrate_step(t, acc, Rwb)
                return
            self.kf.predict(acc, Rwb, t)
            self.yaw = math.radians(data['stabilizer.yaw'])
            self.az_world = float((Rwb @ (acc - self.kf.x[B]))[2] - G)
            self._az_window.append(self.az_world)

            self._acc_window.append(acc)
            self._n_imu += 1
            self._adapt_vibration(acc)
            if self.kf.cfg.drag_k > 0 and not self.motors_off and not self.aiding:
                self._drag_buf.append(acc)
                if len(self._drag_buf) >= self.kf.cfg.drag_every:
                    self.kf.update_drag(np.mean(self._drag_buf, axis=0), Rwb)
                    self._drag_buf.clear()
            if self.imu_history is not None:
                self.imu_history.append((t, *acc, data['stabilizer.roll'],
                                         data['stabilizer.pitch'], data['stabilizer.yaw']))
            self.still = (self.motors_off and len(self._acc_window) == self.ZUPT_WINDOW
                          and np.std(self._acc_window, axis=0).max() < self.ZUPT_ACC_STD)
            if self.still and self._landed_pending:
                self.kf.shock()
                if self.floor_z is not None:
                    self.kf.update_height(self.floor_z)
                self._landed_pending = False
                print(f"\n[kf] landed — velocity reset, z set to floor {self.floor_z}")
            if self.still and self._n_imu % self.ZUPT_EVERY == 0:
                self.kf.update_zero_velocity()
            if self.history is not None:
                self.history.append((t, *self.kf.x[P], *self.kf.uncertainty(),
                                     float(self.aiding), CONTROLLERS.index(self.controller),
                                     self.visual_fixes))

    def aux_callback(self, timestamp, data, _logconf):
        with self.lock:
            motors_off = data['stabilizer.thrust'] < 1.0
            if motors_off and not self.motors_off and self.initialized:
                self._landed_pending = True
            self.motors_off = motors_off
            self.thrust = data['stabilizer.thrust']
            if not self.motors_off:
                self._acc_window.clear()
            self.onboard_p = np.array([data['stateEstimate.x'], data['stateEstimate.y'],
                                       data['stateEstimate.z']])
            if self.aiding and self.initialized and not self.imu_only:
                if self.kf.fix_position(self.onboard_p, self.REF_SIGMA_P):
                    self.visual_fixes += 1
                    self._rej_streak = 0
                else:
                    # the KF is more confident than it should be (motors-off
                    # noise model in flight) and has drifted off a reference it
                    # can't accept any more: re-anchor rather than diverge
                    self._rej_streak += 1
                    if self._rej_streak >= self.REANCHOR_AFTER:
                        # p AND v: re-anchoring p alone keeps the wrong velocity
                        # and it runs off again (flight3: every ~1.4 s)
                        self.kf.fix_position(self.onboard_p, self.REF_SIGMA_P, force=True)
                        if self.onboard_v is not None:
                            self.kf.fix_velocity(self.onboard_v, self.REF_SIGMA_V, force=True)
                        self.reanchors += 1
                        self.visual_fixes += 1
                        self._rej_streak = 0
            if self.ref_history is not None:
                self.ref_history.append((timestamp / 1000.0, data['stateEstimate.x'],
                                         data['stateEstimate.y'], data['stateEstimate.z'],
                                         data['stabilizer.thrust']))

    def _adapt_vibration(self, acc):
        """Motors on: white accel noise = high-passed accel std over the last
        second (never below the Allan value). The motors-off Allan fit misses
        the propeller vibration, which made the KF ~10x overconfident in flight.
        Motors off: back to the Allan value."""
        if self.motors_off:
            self._vib.clear()
            self.kf.cfg.acc_vrw = self.base_vrw
            return
        self._vib.append(acc)
        if len(self._vib) == self.VIB_WINDOW and self._n_imu % 50 == 0:
            a = np.array(self._vib)
            k = np.ones(10) / 10                     # 0.1 s moving average = the real motion
            smooth = np.stack([np.convolve(a[:, j], k, mode='same') for j in range(3)], axis=1)
            sigma = float(np.std((a - smooth)[10:-10], axis=0).max())
            dt = 0.01                                # KF_IMU is logged at 100 Hz
            self.kf.cfg.acc_vrw = max(self.base_vrw, sigma * math.sqrt(dt))

    def refv_callback(self, timestamp, data, _logconf):
        with self.lock:
            self.onboard_v = np.array([data['stateEstimate.vx'], data['stateEstimate.vy'],
                                       data['stateEstimate.vz']])
            if self.aiding and self.initialized and not self.imu_only:
                if self.kf.fix_velocity(self.onboard_v, self.REF_SIGMA_V):
                    self.visual_fixes += 1

    def onboard(self):
        """(onboard p, onboard v), either None until the first packet."""
        with self.lock:
            return (None if self.onboard_p is None else self.onboard_p.copy(),
                    None if self.onboard_v is None else self.onboard_v.copy())

    def lock_imu_only(self):
        """Until release_imu_only() no visual measurement reaches the KF: the
        Flow deck callbacks skip every update and handover() refuses. The
        onboard estimate is still logged (ref_history) for the report."""
        with self.lock:
            self.aiding = False
            self.imu_only = True

    def release_imu_only(self):
        """End of the IMU-only segment: visual measurements may reach the KF again."""
        with self.lock:
            self.imu_only = False

    def handover(self):
        """Stop aiding: from here the KF is on the IMU alone. Returns (KF p,
        onboard p) at that instant, after one last fix of both p and v."""
        with self.lock:
            if self.imu_only:
                raise RuntimeError("IMU-only lock is on: no visual fix allowed")
            if self.onboard_p is not None:
                self.kf.fix_position(self.onboard_p, self.REF_SIGMA_P, force=True)
                self.visual_fixes += 1
            if self.onboard_v is not None:
                self.kf.fix_velocity(self.onboard_v, self.REF_SIGMA_V, force=True)
                self.visual_fixes += 1
            self.aiding = False
            return self.kf.x[P].copy(), (None if self.onboard_p is None
                                         else self.onboard_p.copy())

    def _calibrate_step(self, t, acc, Rwb):
        c = self._calib
        if c["t0"] is None:
            c["t0"] = t
        c["acc"].append(acc)
        c["R"].append(Rwb)
        if t - c["t0"] < self.CALIB_TIME or len(c["acc"]) < 20:
            return

        acc_arr = np.array(c["acc"])
        if acc_arr.std(axis=0).max() > 0.3:   # someone bumped it — start over
            print(f"[kf] calibration: drone moving (accel std "
                  f"{acc_arr.std(axis=0).max():.2f}), retrying")
            self._calib = {"acc": [], "R": [], "t0": None}
            return

        bias = calibrate_bias(c["acc"], c["R"])
        self.kf.reset(self.origin, bias, t)
        self.initialized = True
        print(f"[kf] ready at {self.origin} — accel bias {np.round(bias, 3)} m/s^2 "
              f"(residual {np.linalg.norm(bias):.3f})")

    # ── known spots ─────────────────────────────────────────────────────────
    def fix_position(self, p, sigma=None, force=False):
        """The drone sits at a known spot (start pad, landing pad)."""
        with self.lock:
            if not self.initialized:
                print("[kf] fix ignored: still calibrating")
                return False
            before = self.kf.x[P].copy()
            ok = self.kf.fix_position(p, sigma, force)
            if ok:
                print(f"[kf] fix {np.round(p, 3)} accepted, corrected by "
                      f"{np.round(self.kf.x[P] - before, 3)} m")
            else:
                print(f"[kf] fix {np.round(p, 3)} REJECTED (NIS {self.kf.last_nis:.0f}, "
                      f"estimate {np.round(before, 3)} ± {np.round(self.kf.uncertainty(), 3)})"
                      f" — use force=True if you're sure")
            return ok

    # ── public API ──────────────────────────────────────────────────────────
    def state(self):
        """(z, vz) — same interface as ZEstimator / KalmanState."""
        with self.lock:
            return float(self.kf.x[2]), float(self.kf.x[5])

    def position(self):
        with self.lock:
            return self.kf.x[P].copy()

    def velocity(self):
        with self.lock:
            return self.kf.x[V].copy()

    def uncertainty(self):
        with self.lock:
            return self.kf.uncertainty()

    def full_state(self):
        with self.lock:
            return self.kf.x.copy(), self.kf.P.copy()

    def control_state(self, source='imu'):
        """Everything the controller needs, read under one lock:
        (p, v, yaw, vertical accel over the last 0.1 s, over the last 0.2 s,
        onboard p or None).

        source picks where p, v come from (see SOURCES):
            'imu', 'both'   this KF (with or without visual aiding — that's
                            the `aiding` flag, set by the flight function)
            'visual'        the onboard (Flow deck) estimate"""
        with self.lock:
            w = list(self._az_window)
            az_short = float(np.mean(w[-10:])) if w else 0.0
            az_mean = float(np.mean(w)) if w else 0.0
            onboard_p = None if self.onboard_p is None else self.onboard_p.copy()
            if source == 'visual':
                if self.onboard_p is None or self.onboard_v is None:
                    raise RuntimeError("no onboard estimate yet")
                p, v = self.onboard_p.copy(), self.onboard_v.copy()
            else:
                p, v = self.kf.x[P].copy(), self.kf.x[V].copy()
            return p, v, self.yaw, az_short, az_mean, onboard_p

    def ground_lock(self):
        """On the floor with motors spinning (spin-up): v = 0, z = floor."""
        with self.lock:
            self.kf.update_zero_velocity()
            if self.floor_z is not None:
                self.kf.update_height(self.floor_z)

    def diagnostics(self):
        with self.lock:
            return dict(self.kf.stats)

    def start_logging(self, scf):
        from cflib.crazyflie.log import LogConfig
        confs = []
        for name, period, variables, cb in (
                ('KF_IMU', 10, ('acc.x', 'acc.y', 'acc.z', 'stabilizer.roll',
                                'stabilizer.pitch', 'stabilizer.yaw'), self.imu_callback),
                ('KF_Aux', 20, ('stabilizer.thrust', 'stateEstimate.x', 'stateEstimate.y',
                                'stateEstimate.z'), self.aux_callback),
                ('KF_RefV', 20, ('stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz'),
                 self.refv_callback)):
            conf = LogConfig(name=name, period_in_ms=period)
            for v in variables:
                conf.add_variable(v, 'float')
            scf.cf.log.add_config(conf)
            conf.data_received_cb.add_callback(cb)
            conf.start()
            confs.append(conf)
        return tuple(confs)


# ─────────────────────────────────────────────────────────────────────────────
# Offline check: errors drawn from the same model as the filter's config.
# 3 s on the ground (calibration + zero velocity), a flight, landing, 3 s on
# the ground. Reports the error growth and whether the filter's uncertainty
# is honest (fraction of time the true error is inside its 3-sigma).
#   python kf_crazyflie.py --sim [--plot] [--noise imu_noise.json]
# ─────────────────────────────────────────────────────────────────────────────

def simulate(seed=0, cfg_true: KFConfig | None = None, cfg_filter: KFConfig | None = None,
             flight=20.0, plot=False, verbose=True):
    rng = np.random.default_rng(seed)
    ct = cfg_true or KFConfig()
    cf = cfg_filter or ct
    dt, ground = 0.01, 3.0
    t = np.arange(0.0, flight + 2 * ground, dt)
    s = np.clip((t - ground) / flight, 0.0, 1.0)           # 0..1 over the flight
    env = np.sin(np.pi * s) ** 2                           # takeoff / land smoothly
    p_true = np.stack([0.5 * np.sin(2 * np.pi * s) * env,
                       0.5 * (1 - np.cos(2 * np.pi * s)) * env,
                       1.0 * env], axis=1)
    v_true = np.gradient(p_true, dt, axis=0)
    a_true = np.gradient(v_true, dt, axis=0)
    flying = (t > ground) & (t < ground + flight)

    def gm_step(xk, sigma, tau, n):
        phi = math.exp(-dt / tau)
        return phi * xk + rng.normal(0, sigma * math.sqrt(1 - phi ** 2), n)

    b = rng.normal(0, 0.1, 3)                               # true bias, unknown to the filter
    n = rng.normal(0, ct.gm_sigma, 3)
    tilt = rng.normal(0, ct.tilt_sigma / G, 2)              # attitude error, rad
    kf = InertialKF(cf)
    calib_acc, calib_R = [], []
    est = np.zeros_like(p_true)
    sig = np.zeros_like(p_true)
    for i, ti in enumerate(t):
        b = b + rng.normal(0, ct.bias_rw * math.sqrt(dt), 3)
        n = gm_step(n, ct.gm_sigma, ct.gm_tau, 3)
        tilt = gm_step(tilt, ct.tilt_sigma / G, ct.tilt_tau, 2)
        yaw = 0.3 * np.pi * s[i]
        R_true = rotation(0.0, 0.0, yaw)
        R_meas = rotation(tilt[0], tilt[1], yaw)
        acc = (R_true.T @ (a_true[i] + GRAVITY) + b + n
               + rng.normal(0, ct.acc_vrw / math.sqrt(dt), 3))

        if ti < 1.5:                                        # calibration
            calib_acc.append(acc)
            calib_R.append(R_meas)
            est[i], sig[i] = p_true[0], 0.0
            continue
        if kf.t is None:
            kf.reset(p_true[0], calibrate_bias(calib_acc, calib_R), ti)
        kf.predict(acc, R_meas, ti)
        if not flying[i] and i % 5 == 0:
            kf.update_zero_velocity()
        est[i], sig[i] = kf.x[P], kf.uncertainty()

    err = est - p_true
    inside = np.all(np.abs(err[t >= 1.5]) <= 3 * sig[t >= 1.5] + 1e-9, axis=1).mean()
    i_land = np.searchsorted(t, ground + flight)
    res = {"err_land": err[i_land], "sig_land": sig[i_land],
           "err_end": err[-1], "inside_3sigma": inside}
    if verbose:
        for tt in (5.0, 10.0, flight):
            k = np.searchsorted(t, ground + tt)
            print(f"  t+{tt:4.1f}s  |err| xy {np.linalg.norm(err[k, :2]) * 100:6.1f} cm  "
                  f"z {abs(err[k, 2]) * 100:6.1f} cm   1-sigma xy "
                  f"{np.linalg.norm(sig[k, :2]) * 100:6.1f} z {sig[k, 2] * 100:6.1f} cm")
        print(f"  true error inside 3-sigma: {inside * 100:.0f} % of the time")
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
        for j, name in enumerate('xyz'):
            ax[j].plot(t, p_true[:, j], 'k', label='true')
            ax[j].plot(t, est[:, j], label='KF')
            ax[j].fill_between(t, est[:, j] - 3 * sig[:, j], est[:, j] + 3 * sig[:, j],
                               alpha=0.2, label='3-sigma')
            ax[j].set_ylabel(f'{name} [m]')
        ax[0].legend()
        ax[2].set_xlabel('t [s]')
        plt.tight_layout()
        plt.savefig("kf_sim.png", dpi=200)
        plt.show()
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Hardware flight, reference mode: takeoff, hover, land flown by the ONBOARD
# estimator (Flow deck) through the high-level commander. This KF runs
# alongside and is logged against the onboard estimate — a test of the noise
# model without trusting it with the drone.
#
#   python kf_crazyflie.py --fly-onboard [--noise imu_noise.json] [--height 0.5]
#                                        [--hover 10] [--out flight.npz] [--plot]
#
# Sequence: connect -> Flow deck check -> reset onboard EKF -> KF calibration
# (still, motors off) -> settle with zero-velocity updates -> arm -> takeoff
# -> hover -> land -> stay on the ground (zero velocity again) -> save.
# ─────────────────────────────────────────────────────────────────────────────

def fly_onboard(scf, estimator, height=0.5, hover=10.0, settle=2.0, takeoff_time=2.0):
    cf = scf.cf
    if int(cf.param.get_value("deck.bcFlow2")) != 1:
        raise RuntimeError("Flow deck not detected — the onboard estimator can't hold position")
    try:   # older firmware needs the high-level commander enabled explicitly
        cf.param.set_value("commander.enHighLevel", "1")
    except Exception:
        pass

    print("[fly] resetting onboard estimator — keep the drone still")
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(1.0)

    estimator.history, estimator.ref_history, estimator.imu_history = [], [], []
    logs = estimator.start_logging(scf)
    events = {}
    try:
        while not estimator.initialized:
            time.sleep(0.05)
        time.sleep(settle)                  # zero-velocity updates refine the bias
        if not estimator.still:
            raise RuntimeError("drone not still after calibration — not taking off")

        cf.supervisor.send_arming_request(True)
        time.sleep(0.5)

        events["takeoff"] = time.time()
        print(f"[fly] takeoff to {height} m")
        cf.high_level_commander.takeoff(height, takeoff_time)
        estimator.controller = 'visual'
        time.sleep(takeoff_time + 0.5)

        events["hover"] = time.time()
        t_end = time.time() + hover
        while time.time() < t_end:
            p, sg = estimator.position(), estimator.uncertainty()
            print(f"\r[fly] hover  KF p={np.round(p, 2)} ±{np.round(sg, 2)} m   ",
                  end='', flush=True)
            time.sleep(0.2)
        print()

        events["land"] = time.time()
        print("[fly] landing")
        cf.high_level_commander.land(0.0, takeoff_time)
        time.sleep(takeoff_time + 0.5)
        cf.high_level_commander.stop()
        estimator.controller = 'none'
        events["landed"] = time.time()
        time.sleep(3.0)                     # on the ground: zero velocity stops the drift
    finally:
        try:
            cf.high_level_commander.stop()
        finally:
            cf.commander.send_stop_setpoint()
            for log in logs:
                log.stop()
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Hardware flight, KF-only: the position loop runs here on this KF's estimate
# and sends roll / pitch / yaw-rate / thrust. The drone's own attitude loop
# stabilises; no onboard position or velocity estimate is used for control.
#
#   python kf_crazyflie.py --fly [--source imu|visual|both] [--noise imu_noise.json]
#       [--height 0.5] [--hover 10] [--hover-thrust 38000] [--z-only] [--fence 1.0] [--plot]
#
# --source (SOURCES): imu = this KF alone (the default, and what the rest of
# this comment is about), visual = the onboard Flow deck estimate, both = this
# KF fused with it. Same controller, takeoff and landing logic for all three,
# so flights differ only in the estimate the loop closes on.
#
# The drone WILL drift physically: it holds the position the KF believes in.
#
# Takeoff: the hover thrust isn't known, so thrust ramps up from 70 % of the
# guess, with the KF locked to the floor (v = 0, z = floor) while the
# vertical acceleration says the floor still carries it (on the floor the
# contact force holds it at 0). The lock lets go as soon as it rises above
# ground_acc, so the first centimetres of climb are integrated, not erased.
# Liftoff is confirmed when it averages above liftoff_acc over 0.2 s; then
# a = g (T / T_hover - 1) gives the hover thrust directly and the position
# loop takes over.
#
# Landing: the setpoint descends. The KF's z can't be trusted to say "at 0",
# so touchdown comes from the IMU: either the velocity step of contact
# (upward acceleration averaged over 0.1 s — sitting on the floor and
# descending at constant speed both read 0, only the transition shows), or
# the floor carrying it — thrust well below hover while the vertical
# acceleration stays ~0 (in the air that thrust would make it sink at
# >1 m/s^2). If the setpoint reaches the KF's floor with no contact felt,
# the KF's z has drifted low and the drone is still in the air: it
# switches to the sink below.
#
# Sink (end of landing, and abort): level attitude, descend at sink_vz on
# the KF's VELOCITY, thrust capped below the hover thrust measured at
# liftoff, so it goes down even if the KF velocity has drifted (the KF can
# only slow the descent, never stop it). Same touchdown detection, motors
# cut after sink_timeout. Before liftoff an abort just stops the motors.
# Ctrl-C a second time cuts the motors immediately. Triggers:
#   - Ctrl-C
#   - position error beyond MAX_POS_ERR (the loop isn't holding — wrong
#     sign, wrong hover thrust)
#   - geofence on the ONBOARD estimate, only if a Flow deck is fitted. It is
#     a kill switch, never a control input.
#
# Attitude convention: cflib's send_setpoint takes right-handed angles
# (it negates pitch for the firmware's legacy convention), so +pitch
# accelerates +x and +roll accelerates -y. Do a --z-only flight first.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ControlConfig:
    kp_z: float = 4.0               # 1/s^2
    kd_z: float = 3.0               # 1/s
    ki_z: float = 1.0               # 1/s^3
    i_max: float = 2.0              # m/s^2, integrator clamp
    kp_xy: float = 2.0
    kd_xy: float = 2.0
    max_tilt_deg: float = 8.0
    thrust_min: int = 10001
    thrust_max: int = 60000
    rate_hz: float = 100.0

    takeoff_ramp: float = 3000.0    # thrust units / s
    liftoff_acc: float = 0.4        # m/s^2, mean vertical accel over 0.2 s = airborne
    ground_acc: float = 0.12        # m/s^2, below this the floor is still holding it
    climb_rate: float = 0.25        # m/s
    land_rate: float = 0.2          # m/s
    touchdown_acc: float = 1.0      # m/s^2 over 0.1 s: the velocity step of contact

    floor_thrust: float = 0.93      # x hover thrust: below this with az ~ 0 = on the floor
                                    # (in the air it would sink at >0.7 m/s^2; must be above
                                    # the sink's ~0.91 or a correct velocity never triggers it)
    floor_time: float = 0.3         # s it has to last

    max_pos_err: float = 0.4        # m, KF error from setpoint -> abort
    sink_vz: float = -0.3           # m/s, final descent and abort
    sink_thrust_max: float = 0.97   # x hover thrust: always sinks, whatever the KF says
    sink_timeout: float = 12.0      # s


class PositionController:
    def __init__(self, hover_thrust, cfg: ControlConfig | None = None, xy=True):
        self.cfg = cfg or ControlConfig()
        self.hover_thrust = float(hover_thrust)
        self.xy = xy
        self.i_z = 0.0

    def reset_integrator(self):
        self.i_z = 0.0

    def compute(self, p_ref, v_ref, p, v, yaw, dt):
        """-> (roll_deg, pitch_deg, thrust) for cflib send_setpoint, plus world accel cmd."""
        c = self.cfg
        e_p, e_v = p_ref - p, v_ref - v

        self.i_z = float(np.clip(self.i_z + c.ki_z * e_p[2] * dt, -c.i_max, c.i_max))
        a = np.zeros(3)
        a[2] = c.kp_z * e_p[2] + c.kd_z * e_v[2] + self.i_z
        if self.xy:
            a[:2] = c.kp_xy * e_p[:2] + c.kd_xy * e_v[:2]

        # world -> body (yaw only), then small-angle tilt from desired accel
        cy, sy = math.cos(yaw), math.sin(yaw)
        ax_b = cy * a[0] + sy * a[1]
        ay_b = -sy * a[0] + cy * a[1]
        tmax = math.radians(c.max_tilt_deg)
        pitch = float(np.clip(math.atan2(ax_b, G), -tmax, tmax))
        roll = float(np.clip(-math.atan2(ay_b, G), -tmax, tmax))

        thrust = self.hover_thrust * (1.0 + a[2] / G) / (math.cos(roll) * math.cos(pitch))
        thrust = int(np.clip(thrust, max(c.thrust_min, 0.5 * self.hover_thrust),
                             min(c.thrust_max, 1.5 * self.hover_thrust)))
        return math.degrees(roll), math.degrees(pitch), thrust, a


class TouchdownDetector:
    """Touchdown from the IMU: the KF's z drifts, so "z = floor" can't be trusted.
    Either the velocity step of contact (vertical accel averaged over 0.1 s
    above touchdown_acc), or the floor carrying it: thrust below floor_thrust x
    hover while the vertical accel stays ~0, for floor_time (in the air that
    thrust would sink it at >0.7 m/s^2)."""

    def __init__(self, c: ControlConfig, hover_thrust):
        self.c, self.hover_thrust = c, hover_thrust
        self.t_floor = 0.0

    def reset(self):
        self.t_floor = 0.0

    def __call__(self, az_short, az_mean, thrust, dt):
        c = self.c
        if az_short > c.touchdown_acc:
            return True
        carried = thrust < c.floor_thrust * self.hover_thrust and abs(az_mean) < c.ground_acc
        self.t_floor = self.t_floor + dt if carried else 0.0
        return self.t_floor >= c.floor_time


def sink_on_kf(cf, estimator, hover_thrust, c: ControlConfig, source='imu', tag='[fly]'):
    """Level descent at sink_vz on the estimate's vertical velocity, thrust
    capped at sink_thrust_max x hover so it always goes down (the estimate can
    only slow the descent, never stop it), until touchdown or sink_timeout.
    Returns True on touchdown. The caller stops the motors."""
    dt = 1.0 / c.rate_hz
    touched = TouchdownDetector(c, hover_thrust)
    thrust = hover_thrust
    t_end = time.time() + c.sink_timeout
    while time.time() < t_end:
        _, v, _, az_short, az_mean, _ = estimator.control_state(source)
        if touched(az_short, az_mean, thrust, dt):
            print(f"{tag} touchdown")
            return True
        thrust = float(np.clip(hover_thrust * (1.0 + c.kd_z * (c.sink_vz - v[2]) / G),
                               0.75 * hover_thrust, c.sink_thrust_max * hover_thrust))
        cf.commander.send_setpoint(0, 0, 0, int(thrust))
        time.sleep(dt)
    print(f"{tag} no touchdown detected — cutting motors")
    return False


class Abort(Exception):
    pass


def fly(scf, estimator, height=0.5, hover=10.0, hover_thrust=38000, xy=True, fence=1.0,
        ctrl: ControlConfig | None = None, settle=2.0, source='imu'):
    """Takeoff, hover, land with the host position loop on `source` (SOURCES)."""
    cf = scf.cf
    c = ctrl or ControlConfig()
    dt = 1.0 / c.rate_hz
    has_deck = int(cf.param.get_value("deck.bcFlow2")) == 1
    if source != 'imu' and not has_deck:
        raise RuntimeError(f"--source {source} needs the Flow deck")
    print(f"[fly] flying on {SOURCE_LABEL[source]}")
    if has_deck:
        print(f"[fly] Flow deck fitted — onboard estimate used as a {fence} m geofence"
              + (" only" if source == 'imu' else ""))
        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(1.0)
    else:
        print("[fly] no Flow deck — no independent geofence, only the KF's own error check")

    estimator.history, estimator.ref_history, estimator.imu_history = [], [], []
    logs = estimator.start_logging(scf)
    events = {}
    last_cmd = [0.0, 0.0, 0]
    ctl = None
    detector = None

    def touched_down(az_short, az_mean, thrust):
        return detector(az_short, az_mean, thrust, dt)

    def send(roll, pitch, thrust):
        last_cmd[:] = [roll, pitch, thrust]
        cf.commander.send_setpoint(roll, pitch, 0.0, int(thrust))

    def sink():
        return sink_on_kf(cf, estimator, ctl.hover_thrust, c, source)

    def check_fence(onboard_p):
        if has_deck and onboard_p is not None and (
                np.linalg.norm(onboard_p[:2]) > fence or onboard_p[2] > fence + height):
            raise Abort(f"geofence: onboard estimate {np.round(onboard_p, 2)}")

    try:
        while not estimator.initialized:
            time.sleep(0.05)
        estimator.aiding = source == 'both'
        time.sleep(settle)
        if not estimator.still:
            raise RuntimeError("drone not still after calibration — not taking off")
        p0 = estimator.control_state(source)[0]

        cf.supervisor.send_arming_request(True)
        time.sleep(0.5)
        send(0, 0, 0)                                    # unlock the commander
        estimator.controller = host_controller(source)
        time.sleep(0.1)

        try:
            # ── takeoff: find the hover thrust ─────────────────────────────
            print("[fly] spinning up, looking for liftoff")
            events["spinup"] = time.time()
            thrust = 0.7 * hover_thrust
            while True:
                p, v, yaw, _, az_mean, onboard_p = estimator.control_state(source)
                check_fence(onboard_p)
                if az_mean > c.liftoff_acc:
                    break
                if az_mean < c.ground_acc:   # floor still carries it; once it doesn't,
                    estimator.ground_lock()  # let the KF integrate the first climb
                thrust += c.takeoff_ramp * dt
                if thrust > 1.4 * hover_thrust:
                    raise Abort(f"no liftoff by thrust {thrust:.0f} — raise --hover-thrust?")
                send(0, 0, thrust)
                time.sleep(dt)
            hover_est = thrust / (1.0 + az_mean / G)
            ctl = PositionController(hover_est, c, xy=xy)
            detector = TouchdownDetector(c, hover_est)
            print(f"[fly] liftoff at thrust {thrust:.0f} -> hover thrust {hover_est:.0f}")
            events["liftoff"] = time.time()

            # ── climb, hover, descend: one setpoint profile ─────────────────
            p_ref = p0.copy()
            p_ref[2] = estimator.control_state(source)[0][2]
            z_hover = p0[2] + height
            phase, t_phase = 'climb', time.time()
            while True:
                p, v, yaw, az, az_mean, onboard_p = estimator.control_state(source)
                check_fence(onboard_p)
                v_ref = np.zeros(3)
                if phase == 'climb':
                    p_ref[2] = min(p_ref[2] + c.climb_rate * dt, z_hover)
                    v_ref[2] = c.climb_rate if p_ref[2] < z_hover else 0.0
                    if p_ref[2] >= z_hover and abs(p[2] - z_hover) < 0.05:
                        phase, t_phase = 'hover', time.time()
                        events["hover"] = t_phase
                        print(f"[fly] hovering {hover:.0f} s at KF z = {z_hover:.2f} m")
                elif phase == 'hover':
                    if time.time() - t_phase >= hover:
                        phase, t_phase = 'land', time.time()
                        events["land"] = t_phase
                        print("[fly] landing")
                elif phase == 'land':
                    p_ref[2] -= c.land_rate * dt
                    v_ref[2] = -c.land_rate
                    if touched_down(az, az_mean, last_cmd[2]):
                        print(f"[fly] touchdown (KF z = {p[2]:+.2f} m)")
                        break
                    if p_ref[2] <= p0[2]:
                        # the KF thinks it's down but felt no contact: its z has
                        # drifted, so sink until the IMU feels the floor
                        print("[fly] estimate says floor, no contact — sinking")
                        sink()
                        break

                err = np.linalg.norm((p_ref - p) if xy else (p_ref - p)[2:])
                if err > c.max_pos_err:
                    raise Abort(f"KF position error {err:.2f} m from setpoint in {phase}")

                roll, pitch, thrust, _ = ctl.compute(p_ref, v_ref, p, v, yaw, dt)
                send(roll, pitch, thrust)
                time.sleep(dt)
            events["landed"] = time.time()

        except (Abort, KeyboardInterrupt) as exc:
            events["abort"] = time.time()
            if ctl is None:
                print(f"\n[fly] ABORT before liftoff: {exc or 'Ctrl-C'} — motors off")
            else:
                print(f"\n[fly] ABORT: {exc or 'Ctrl-C'} — level descent at "
                      f"{c.sink_vz} m/s (Ctrl-C again to cut motors)")
                try:
                    sink()
                except KeyboardInterrupt:
                    print("[fly] motors cut")

        send(0, 0, 0)
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()
        estimator.controller = 'none'
        time.sleep(3.0)                     # on the ground: zero velocity again
    finally:
        cf.commander.send_stop_setpoint()
        for log in logs:
            log.stop()
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Hardware flight, handover: the onboard ("visual", Flow deck) estimator
# takes off and hovers; this KF then flies the drone `distance` m forward on
# the IMU alone; the onboard estimator takes back over and lands.
#
#   python kf_crazyflie.py --handover [--source imu|visual|both] [--noise imu_noise.json]
#       [--height 0.5] [--distance 1.0] [--move-time 4] [--hold 2] [--max-dev 0.5] [--plot]
#
# --source picks what step 3 flies on (SOURCES). With visual or both the only
# reference is the onboard estimate the controller itself uses, so the report
# can show tracking, not accuracy — that needs mocap.
#
#   1. ground   calibrate the KF, settle (zero velocity), arm. One zero-thrust
#               attitude setpoint unlocks the firmware's thrust lock — it
#               must happen here: mid-air, that packet would cut the motors.
#               Then notify-stop so the high-level commander can take over.
#   0. checks   the onboard estimate must hold still on the ground (else the
#               Flow deck isn't working and the visual takeoff would chase a
#               drifting estimate) — refuses to arm otherwise, and checks
#               again right before the takeoff command. In every visual phase
#               a guard watches it: faster than 1.5 m/s or outside the flight
#               area -> VisualFault -> stop fusing it, cut the motors if
#               below 30 cm, else a level descent on the KF until touchdown.
#   2. visual   high-level takeoff, then WAIT until the planned takeoff time has
#               passed and the onboard z is within 15 cm of the target with
#               its vertical speed < 5 cm/s, for 1 s; hover `aided` s. The KF
#               fuses the onboard estimate the whole time, so it learns bias
#               and tilt error with the motors running; the logged thrust
#               gives the hover thrust. Handover only if the height is steady
#               and the KF agrees with the onboard estimate within 15 cm —
#               otherwise it lands on visual without handing over.
#   3. KF       handover: the KF starts from the onboard p, v and stops fusing.
#               Its position loop sends roll / pitch / thrust and flies a
#               min-jerk move of `distance` m along +x, then holds `hold` s.
#      imu      during step 3 the KF is locked IMU-only (lock_imu_only): the
#               Flow deck callbacks can't update it, and no decision uses the
#               onboard estimate — it is only logged, to measure the KF's
#               error afterwards. A failure can only come from the KF itself
#               (far from its setpoint) or Ctrl-C.
#   4. visual   at the end of the move, or on a failure: notify-stop hands
#               back; the high-level commander (planning from the onboard
#               state) goes to where the drone really is, then lands. The
#               IMU-only lock is released and aiding resumes.
#
# In step 3 anything wrong hands straight back to visual control: Ctrl-C,
# the KF more than max_pos_err from its setpoint, or the onboard estimate
# more than `max_dev` from where the move should have put it. A second
# Ctrl-C during the hand-back cuts the motors.
#
# The result: how far the drone really moved (onboard) while the KF believed
# it moved `distance`.
# ─────────────────────────────────────────────────────────────────────────────

def min_jerk(t, T):
    """Position fraction and its rate for a min-jerk move of duration T."""
    tau = min(max(t / T, 0.0), 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    return s, ds


class VisualFault(Exception):
    """The onboard (Flow deck) estimate stopped making sense during a visual phase."""


def check_visual_on_ground(cf, estimator, duration=1.5, max_drift=0.05, max_speed=0.05,
                           probe_quality=True):
    """The onboard (Flow deck) estimate must hold still while the drone does.
    If it drifts on the ground, the Flow deck isn't correcting it (featureless
    or shiny floor, poor light, dirty sensor) and a visual takeoff would fly
    off chasing it. Returns the onboard position."""
    from cflib.crazyflie.log import LogConfig
    quality = []
    probe = None
    names = [n for n in ('motion.squal', 'range.zrange')
             if probe_quality and cf.log.toc.get_element_by_complete_name(n) is not None]
    if names:
        probe = LogConfig(name='KF_FlowProbe', period_in_ms=50)
        for n in names:
            probe.add_variable(n, 'uint8_t' if n == 'motion.squal' else 'uint16_t')
        cf.log.add_config(probe)
        probe.data_received_cb.add_callback(lambda ts, d, _: quality.append(d))
        probe.start()

    p_start, _ = estimator.onboard()
    t_end = time.time() + duration
    worst_drift, worst_speed = 0.0, 0.0
    while time.time() < t_end:
        p, v = estimator.onboard()
        if p is not None and p_start is not None:
            worst_drift = max(worst_drift, float(np.linalg.norm(p - p_start)))
        if v is not None:
            worst_speed = max(worst_speed, float(np.linalg.norm(v)))
        time.sleep(0.02)
    if probe is not None:
        probe.stop()

    info = ""
    if quality:
        if 'motion.squal' in quality[0]:
            info += f", flow surface quality {np.median([q['motion.squal'] for q in quality]):.0f}"
        if 'range.zrange' in quality[0]:
            info += f", ToF range {np.median([q['range.zrange'] for q in quality]):.0f} mm"
    print(f"[handover] visual check on the ground: drift {worst_drift * 100:.1f} cm, "
          f"speed {worst_speed:.3f} m/s{info}")
    if worst_drift > max_drift or worst_speed > max_speed:
        raise RuntimeError(
            "onboard (Flow deck) estimate drifts while the drone is still — not taking off. "
            "Check the floor under the deck has texture and isn't shiny, the light, and that "
            "the flow sensor lens is clean (low surface quality confirms it).")
    return estimator.onboard()[0]


def visual_guard(estimator, home, radius, max_speed=1.5):
    """Returns a check for visual phases: raises VisualFault if the onboard
    estimate moves implausibly fast or leaves `radius` around `home` (x, y).
    flight_imu.npz: the estimate started drifting on the ground after the
    ground check had passed, and the visual takeoff chased it into a flip."""
    def guard():
        p, v = estimator.onboard()
        if p is None or v is None:
            return
        speed = float(np.linalg.norm(v[:2]))
        if speed > max_speed:
            raise VisualFault(f"onboard estimate moving at {speed:.1f} m/s")
        off = float(np.linalg.norm(p[:2] - home[:2]))
        if off > radius:
            raise VisualFault(f"onboard estimate {off:.1f} m from the takeoff point")
    return guard


def visual_sleep(seconds, guard):
    t_end = time.time() + seconds
    while time.time() < t_end:
        guard()
        time.sleep(0.02)


def emergency_descent(cf, estimator, hover_thrust, c, cut_below=0.3):
    """The visual estimate is broken: stop fusing it, then cut the motors if
    low, else descend level on the KF's vertical velocity with thrust capped
    below hover (it always goes down), until the IMU feels the floor."""
    estimator.aiding = False
    estimator.controller = 'host'
    dt = 1.0 / c.rate_hz
    z = estimator.position()[2]
    if hover_thrust is None or z < cut_below:
        print(f"[handover] EMERGENCY: motors cut (KF z {z:.2f} m)")
        cf.commander.send_stop_setpoint()
        estimator.controller = 'none'
        return
    print(f"[handover] EMERGENCY: level descent on the KF from z {z:.2f} m (Ctrl-C to cut)")
    try:
        sink_on_kf(cf, estimator, hover_thrust, c, tag='[handover]')
    finally:
        print("[handover] motors off")
        cf.commander.send_stop_setpoint()
        estimator.controller = 'none'


def _onboard_vz_avg(estimator, window=0.5):
    """Onboard vertical speed averaged over `window` s (from the logged reference)."""
    ref = estimator.ref_history
    if not ref or len(ref) < 2:
        return None
    t_now = ref[-1][0]
    old = next((r for r in reversed(ref) if t_now - r[0] >= window), ref[0])
    return (ref[-1][3] - old[3]) / max(t_now - old[0], 1e-3)


def wait_hover(estimator, height, t_min, timeout, tol_z=0.15, max_vz=0.05, steady=1.0,
               guard=lambda: None):
    """Takeoff is complete when, after the planned takeoff time t_min, the onboard
    z is within tol_z of `height` and its vertical speed (0.5 s average) is below
    max_vz, for `steady` s. Horizontal speed isn't checked: a Flow-deck hover
    oscillates by a few cm at ~0.2 m/s and that's normal. Returns the hover
    position, or None on timeout."""
    t_start = time.time()
    t_ok = None
    while time.time() - t_start < timeout:
        guard()
        p, _ = estimator.onboard()
        vz = _onboard_vz_avg(estimator)
        ok = (time.time() - t_start >= t_min and p is not None and vz is not None
              and abs(p[2] - height) < tol_z and abs(vz) < max_vz)
        if ok:
            t_ok = t_ok or time.time()
            if time.time() - t_ok >= steady:
                return p
        else:
            t_ok = None
        time.sleep(0.02)
    p, _ = estimator.onboard()
    vz = _onboard_vz_avg(estimator)
    print(f"[handover] takeoff not settled: onboard z {p[2]:.2f} m (target {height}), "
          f"vz {vz if vz is None else round(vz, 3)} m/s")
    return None


def ready_for_handover(estimator, p_hover, timeout, max_vz=0.05, max_offset=0.15,
                       max_disagree=0.15, guard=lambda: None):
    """Steady height, near the hover point, KF and onboard estimate roughly agreeing
    (the handover re-anchors the KF on the onboard state anyway)."""
    t_end = time.time() + timeout
    while True:
        guard()
        p_kf = estimator.position()
        p, _ = estimator.onboard()
        vz = _onboard_vz_avg(estimator) or 0.0
        disagree = float(np.linalg.norm(p_kf - p))
        offset = float(np.linalg.norm(p - p_hover))
        if abs(vz) < max_vz and offset < max_offset and disagree < max_disagree:
            return True
        if time.time() > t_end:
            print(f"[handover] not ready: vz {vz:.2f} m/s, {offset:.2f} m from hover point, "
                  f"KF-onboard {disagree:.2f} m")
            return False
        time.sleep(0.02)


def fly_handover(scf, estimator, height=0.5, distance=1.0, move_time=4.0, hold=2.0,
                 aided=4.0, max_dev=0.5, ctrl: ControlConfig | None = None, settle=2.0,
                 source='imu'):
    cf = scf.cf
    c = ctrl or ControlConfig()
    dt = 1.0 / c.rate_hz
    if int(cf.param.get_value("deck.bcFlow2")) != 1:
        raise RuntimeError("Flow deck not detected — visual takeoff and landing need it")
    try:   # older firmware needs the high-level commander enabled explicitly
        cf.param.set_value("commander.enHighLevel", "1")
    except Exception:
        pass

    print("[handover] resetting onboard estimator — keep the drone still")
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(1.0)

    estimator.history, estimator.ref_history, estimator.imu_history = [], [], []
    logs = estimator.start_logging(scf)
    hl = cf.high_level_commander
    result = {}
    try:
        while not estimator.initialized:
            time.sleep(0.05)
        time.sleep(settle)
        if not estimator.still:
            raise RuntimeError("drone not still after calibration — not taking off")
        home = check_visual_on_ground(cf, estimator)
        estimator.aiding = True

        # ── 1. ground ──────────────────────────────────────────────────────
        cf.supervisor.send_arming_request(True)
        time.sleep(0.5)
        cf.commander.send_setpoint(0, 0, 0, 0)          # unlock thrust, on the ground
        time.sleep(0.1)
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.1)
        # again, right before takeoff: the estimate can start drifting after the
        # first check passed (flight_imu.npz: ~1 s later, 5 m off by takeoff)
        check_visual_on_ground(cf, estimator, duration=0.5, probe_quality=False)
        guard = visual_guard(estimator, home, radius=distance + max_dev + 0.5)

        # ── 2. visual takeoff + aided hover ─────────────────────────────────
        takeoff_time = max(2.0, height / c.climb_rate)
        print(f"[handover] visual takeoff to {height} m ({takeoff_time:.1f} s)")
        thrusts = []
        try:
            hl.takeoff(height, takeoff_time)
            estimator.controller = 'visual'
            p_hover = wait_hover(estimator, height, t_min=takeoff_time,
                                 timeout=takeoff_time + 8.0, guard=guard)
            if p_hover is None:
                raise Abort("takeoff didn't settle at the target height")
            print(f"[handover] takeoff complete at {np.round(p_hover, 2)} — aided hover {aided:.0f} s")

            t_end = time.time() + aided
            while time.time() < t_end:
                guard()
                thrusts.append(estimator.thrust)
                time.sleep(0.02)
            if not ready_for_handover(estimator, p_hover, timeout=3.0, guard=guard):
                raise Abort("not steady, or KF not agreeing with the onboard estimate")
        except VisualFault as exc:
            print(f"\n[handover] VISUAL FAULT: {exc}")
            emergency_descent(cf, estimator,
                              float(np.median(thrusts)) if len(thrusts) > 25 else None, c)
            return result
        except (Abort, KeyboardInterrupt) as exc:
            print(f"\n[handover] {exc or 'Ctrl-C'} — no handover, visual landing")
            hl.land(0.0, max(2.5, height / c.land_rate))
            time.sleep(max(2.5, height / c.land_rate) + 0.5)
            hl.stop()
            estimator.controller = 'none'
            return result
        hover_thrust = float(np.median(thrusts[len(thrusts) // 2:]))
        p_kf, _, _, _, _, p_on = estimator.control_state()
        print(f"[handover] ready: KF {np.round(p_kf, 3)} vs onboard {np.round(p_on, 3)}, "
              f"hover thrust {hover_thrust:.0f}, re-anchors {estimator.reanchors}, "
              f"in-flight acc_vrw {estimator.kf.cfg.acc_vrw:.4f} "
              f"(Allan {estimator.base_vrw:.4f})")

        # ── 3. the host controller flies, on `source` ───────────────────────
        ctl = PositionController(hover_thrust, c, xy=True)
        p0_kf, p0_on = estimator.handover()      # KF starts from the onboard p, v (last visual read)
        if source == 'imu':
            estimator.lock_imu_only()            # no visual measurement reaches the KF again
        else:
            estimator.aiding = True              # visual keeps correcting the KF
        fixes0 = estimator.visual_fixes          # after the handover snap: must not move for 'imu'
        estimator.controller = host_controller(source)
        p0_ctl = estimator.control_state(source)[0]
        direction = np.array([1.0, 0.0, 0.0])
        result.update(source=source, p0_ctl=p0_ctl, p0_on=p0_on, t_handover=time.time())
        print(f"[handover] {SOURCE_LABEL[source]} in control — moving {distance} m along +x")
        reason = None
        try:
            t0 = time.time()
            while True:
                t = time.time() - t0
                if t > move_time + hold:
                    break
                s, ds = min_jerk(t, move_time)
                p_ref = p0_ctl + direction * distance * s
                v_ref = direction * distance * ds
                p, v, yaw, _, _, p_on = estimator.control_state(source)

                err = np.linalg.norm(p_ref - p)
                if err > c.max_pos_err:
                    reason = f"{SOURCE_LABEL[source]} {err:.2f} m from its setpoint"
                    break
                if source != 'imu':              # imu: no decision may use the Flow deck
                    dev = np.linalg.norm(p_on - (p0_on + direction * distance * s))
                    if dev > max_dev:
                        reason = f"onboard estimate {dev:.2f} m off the planned path"
                        break

                roll, pitch, thrust, _ = ctl.compute(p_ref, v_ref, p, v, yaw, dt)
                cf.commander.send_setpoint(roll, pitch, 0.0, thrust)
                if int(t / dt) % 25 == 0:
                    print(f"\r[handover] {SOURCE_LABEL[source]}: {t:4.1f} s, aiding "
                          f"{'ON' if estimator.aiding else 'off'}, visual corrections since "
                          f"handover {estimator.visual_fixes - fixes0}   ", end='', flush=True)
                time.sleep(dt)
        except KeyboardInterrupt:
            reason = "Ctrl-C"
        result.update(visual_fixes=estimator.visual_fixes - fixes0)
        result.update(p1_ctl=estimator.control_state(source)[0],
                      p1_on=estimator.control_state()[5],
                      sigma1=None if source == 'visual' else estimator.uncertainty(),
                      t_back=time.time(), reason=reason)

        # ── 4. visual hand-back + landing ───────────────────────────────────
        print(f"\n[handover] back to visual control"
              + (f" — ABORT: {reason}" if reason else "") + " (Ctrl-C to cut motors)")
        try:
            cf.commander.send_notify_setpoint_stop()
            estimator.controller = 'visual'
            estimator.release_imu_only()        # the IMU-only segment is over
            p_on = estimator.control_state()[5]
            hl.go_to(p_on[0], p_on[1], height, estimator.control_state()[2], 1.0)
            visual_sleep(2.0, guard)
            estimator.aiding = True
            print("[handover] visual landing")
            land_time = max(2.5, height / c.land_rate)
            hl.land(0.0, land_time)
            visual_sleep(land_time + 0.5, guard)
            hl.stop()
            estimator.controller = 'none'
            time.sleep(2.0)
        except VisualFault as exc:
            print(f"\n[handover] VISUAL FAULT: {exc}")
            emergency_descent(cf, estimator, hover_thrust, c)
        except KeyboardInterrupt:
            print("[handover] motors cut")
    finally:
        cf.commander.send_stop_setpoint()
        for log in logs:
            log.stop()
    return result


def handover_report(result, distance):
    if "p1_ctl" not in result:
        return
    source = result["source"]
    d_ctl = result["p1_ctl"] - result["p0_ctl"]
    d_on = result["p1_on"] - result["p0_on"]
    print(f"\n[handover] {SOURCE_LABEL[source]} segment, "
          f"{result['t_back'] - result['t_handover']:.1f} s"
          + (f" (cut short: {result['reason']})" if result["reason"] else ""))
    n_vis = result.get("visual_fixes", 0)
    if source == 'imu':
        print(f"  visual corrections applied to the KF during it: {n_vis}"
              + ("  -> IMU only, confirmed" if n_vis == 0 else "  -> NOT IMU only!"))
    else:
        print(f"  visual corrections applied during it: {n_vis} (expected for '{source}')")
    print(f"  commanded          {np.round([distance, 0, 0], 3)} m")
    print(f"  controller thought {np.round(d_ctl, 3)} m")
    print(f"  onboard estimate   {np.round(d_on, 3)} m")
    if source == 'imu':
        print(f"  error              {np.round(d_ctl - d_on, 3)} m  (|xy| "
              f"{np.linalg.norm((d_ctl - d_on)[:2]):.3f}, KF claimed 1-sigma "
              f"{np.round(result['sigma1'], 3)})")
    else:
        # the only reference is the onboard estimate, which this source uses
        # itself: tracking is all that can be measured, not accuracy
        print(f"  tracking error     {np.round(d_ctl - np.array([distance, 0, 0]), 3)} m "
              f"(vs commanded; no independent truth for '{source}' without mocap)")


def _spans(mask):
    """(start, end) index pairs of the True runs in a boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def imu_only_spans(t, aiding, controller, vfix):
    """Spans where the host controller flew with aiding off, each with the
    number of visual corrections applied inside it (0 = IMU only, proven)."""
    mask = (controller == CONTROLLERS.index('host')) & ~aiding
    return [(a, b, int(vfix[b] - vfix[a])) for a, b in _spans(mask) if b > a]


def flight_report(estimator, out=None, plot=False):
    kf_h = np.array(estimator.history)
    ref = np.array(estimator.ref_history)
    if len(kf_h) == 0 or len(ref) == 0:
        print("[fly] nothing recorded")
        return
    t0 = ref[0, 0]
    t_kf, p_kf, s_kf = kf_h[:, 0] - t0, kf_h[:, 1:4], kf_h[:, 4:7]
    aiding, controller, vfix = kf_h[:, 7] > 0.5, kf_h[:, 8].astype(int), kf_h[:, 9]
    t_ref, p_ref, thrust = ref[:, 0] - t0, ref[:, 1:4], ref[:, 4]
    imu_only = imu_only_spans(t_kf, aiding, controller, vfix)
    # onboard reference interpolated onto the KF timeline
    p_on = np.stack([np.interp(t_kf, t_ref, p_ref[:, j]) for j in range(3)], axis=1)
    err = p_kf - p_on
    inside = np.mean(np.all(np.abs(err) <= 3 * s_kf, axis=1)) * 100

    air = np.interp(t_kf, t_ref, (thrust > 1.0).astype(float)) > 0.5
    if air.any():
        k = np.nonzero(air)[0][-1]
        print(f"[fly] at touchdown: KF - onboard = {np.round(err[k], 3)} m, "
              f"KF 1-sigma {np.round(s_kf[k], 3)} m, after {np.ptp(t_kf[air]):.1f} s in the air")
    print(f"[fly] end:          KF - onboard = {np.round(err[-1], 3)} m")
    print(f"[fly] KF error inside its 3-sigma {inside:.0f}% of the time "
          f"(well below ~99% means the noise model is too optimistic)")
    print("[fly]", estimator.diagnostics())
    for a, b, n in imu_only:
        print(f"[fly] flown on the IMU alone: {t_kf[a]:.2f}-{t_kf[b]:.2f} s ({t_kf[b] - t_kf[a]:.1f} s),"
              f" visual corrections in it: {n}" + ("" if n == 0 else "  <-- NOT IMU only"))
    if not imu_only:
        print("[fly] no IMU-only control in this flight")

    if out:
        extra = {}
        if estimator.imu_history:
            imu = np.array(estimator.imu_history)
            extra = dict(t_imu=imu[:, 0] - t0, acc=imu[:, 1:4], rpy_deg=imu[:, 4:7])
        np.savez(out, t_kf=t_kf, p_kf=p_kf, sigma_kf=s_kf, aiding=aiding,
                 controller=controller, visual_fixes=vfix, controllers=np.array(CONTROLLERS),
                 t_ref=t_ref, p_ref=p_ref, thrust=thrust, **extra)
        print(f"[fly] saved to {out}")
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
        for j, name in enumerate('xyz'):
            ax[j].plot(t_ref, p_ref[:, j], 'k', label='onboard (Flow deck)')
            ax[j].plot(t_kf, p_kf[:, j], label='KF')
            ax[j].fill_between(t_kf, p_kf[:, j] - 3 * s_kf[:, j], p_kf[:, j] + 3 * s_kf[:, j],
                               alpha=0.2, label='KF 3-sigma')
            ax[j].set_ylabel(f'{name} [m]')
        for axis in ax:
            for a, b, _ in imu_only:
                axis.axvspan(t_kf[a], t_kf[b], color='tab:orange', alpha=0.15, lw=0)
            vis = np.isin(controller, [CONTROLLERS.index('visual'),
                                       CONTROLLERS.index('host_visual')])
            for a, b in _spans(vis):
                axis.axvspan(t_kf[a], t_kf[b], color='tab:green', alpha=0.08, lw=0)
        ax[0].plot([], [], color='tab:orange', alpha=0.4, lw=6, label='flown on IMU only')
        ax[0].plot([], [], color='tab:green', alpha=0.25, lw=6, label='flown on visual')
        ax[0].legend()
        ax[3].plot(t_ref, thrust)
        ax[3].set_ylabel('thrust')
        ax[3].set_xlabel('t [s]')
        plt.tight_layout()
        if out:
            png = os.path.splitext(out)[0] + '.png'
            fig.savefig(png, dpi=150)
            print(f"[fly] figure saved to {png}")
        plt.show()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', action='store_true', help='offline simulation, no drone')
    ap.add_argument('--fly', action='store_true', help='takeoff, hover, land on this KF only')
    ap.add_argument('--fly-onboard', action='store_true',
                    help='same flight on the onboard estimator, KF logged alongside')
    ap.add_argument('--hover-thrust', type=float, default=38000,
                    help='first guess; the real one is found at liftoff')
    ap.add_argument('--z-only', action='store_true', help='KF controls height only, attitude level')
    ap.add_argument('--source', choices=SOURCES, default='imu',
                    help='what the host controller flies on (--fly, --handover): '
                         'imu = this KF alone, visual = onboard Flow deck estimate, '
                         'both = this KF fused with the Flow deck')
    ap.add_argument('--handover', action='store_true',
                    help='visual takeoff, KF flies --distance forward, visual landing')
    ap.add_argument('--distance', type=float, default=1.0, help='m, KF move along +x')
    ap.add_argument('--move-time', type=float, default=4.0, help='s')
    ap.add_argument('--hold', type=float, default=2.0, help='s on the KF after the move')
    ap.add_argument('--max-dev', type=float, default=0.5,
                    help='m, hand back if the onboard estimate strays this far from the plan '
                         '(--source visual/both only: imu never reads the Flow deck after handover)')
    ap.add_argument('--fence', type=float, default=1.0,
                    help='m, abort if the onboard (Flow deck) estimate leaves this radius')
    ap.add_argument('--noise', help='imu_noise.json from imu_allan.py')
    ap.add_argument('--height', type=float, default=0.5)
    ap.add_argument('--hover', type=float, default=10.0)
    ap.add_argument('--out', default='flight.npz')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    cfg = KFConfig.from_json(args.noise) if args.noise else None
    if cfg:
        print("[kf] noise model:", cfg)
    if args.sim:
        simulate(cfg_true=cfg, plot=args.plot)
    elif args.fly or args.fly_onboard or args.handover:
        URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702')
        logging.basicConfig(level=logging.ERROR)
        cflib.crtp.init_drivers()
        estimator = CrazyflieKF(origin=(0.0, 0.0, 0.0), cfg=cfg)
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            try:
                if args.handover:
                    res = fly_handover(scf, estimator, height=args.height,
                                       distance=args.distance, move_time=args.move_time,
                                       hold=args.hold, max_dev=args.max_dev,
                                       source=args.source)
                    handover_report(res, args.distance)
                elif args.fly:
                    fly(scf, estimator, height=args.height, hover=args.hover,
                        hover_thrust=args.hover_thrust, xy=not args.z_only, fence=args.fence,
                        source=args.source)
                else:
                    fly_onboard(scf, estimator, height=args.height, hover=args.hover)
            except KeyboardInterrupt:
                print("\n[fly] interrupted — motors stopped")
            finally:
                flight_report(estimator, out=args.out, plot=args.plot)
    else:
        ap.print_help()
