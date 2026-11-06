"""
Replay a recorded flight (flight.npz from kf_crazyflie.py, with raw IMU)
through the KF offline:

    python kf_replay.py flight3.npz --fit            fit the in-flight noise model
    python kf_replay.py flight3.npz --coast          IMU-only coasting test
    python kf_replay.py flight3.npz --fit --coast --drag --out imu_noise_flight.json

--fit
    Replays the KF with the onboard estimate fused as position measurements
    and picks the noise parameters (acc_vrw, gm_sigma, gm_tau, tilt_sigma,
    tilt_tau) that maximise the likelihood of the innovations while the
    motors run. A motors-off Allan fit can't see what matters in flight —
    attitude error, thrust/drag effects, propwash — this can.

--coast
    The handover, offline, many times: from every 0.5 s of the aided hover,
    the KF continues on the IMU alone for 2 s and 4 s; its displacement is
    compared with the onboard one. Reports the drift and whether the KF's
    3-sigma actually covers it. This predicts what a KF segment can do before
    flying it.

--drag
    Rotor drag: on a quadrotor the body x/y accelerometer reads mostly drag,
    a_xy = -k v_body_xy (flight3: corr -0.79). Used as a measurement it gives
    an IMU-only handle on velocity. k and the residual are fitted from the
    aided data; the residual is time-correlated, so updates use 10-sample
    means and the residual sigma x DRAG_SIGMA_MULT (flight3: x1 at 100 Hz made
    things worse, x10 at 10 Hz cut the 4 s drift 77 -> 47 cm). Written into
    the output JSON; the coasting test runs with and without it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, replace

import numpy as np

import kf_crazyflie as k

REF_SIGMA = 0.03
DRAG_SIGMA_MULT = 10.0


def load(path):
    d = np.load(path)
    if 't_imu' not in d:
        raise SystemExit(f"{path} has no raw IMU — record it with the current kf_crazyflie.py")
    R = np.array([k.rotation(math.radians(r), k.PITCH_SIGN * math.radians(p), math.radians(y))
                  for r, p, y in d['rpy_deg']])
    thrust_imu = np.interp(d['t_imu'], d['t_ref'], d['thrust'])
    return dict(t=d['t_imu'], acc=d['acc'], R=R, on=thrust_imu > 1.0,
                t_ref=d['t_ref'], p_ref=d['p_ref'], thrust=d['thrust'])


def onboard_velocity(f):
    from scipy.signal import savgol_filter
    dt = float(np.median(np.diff(f['t_ref'])))
    v = np.stack([savgol_filter(f['p_ref'][:, j], 11, 2, deriv=1, delta=dt) for j in range(3)], 1)
    return np.stack([np.interp(f['t'], f['t_ref'], v[:, j]) for j in range(3)], 1)


# ── drag ────────────────────────────────────────────────────────────────────
def fit_drag(f, start):
    """kd, sigma from the aided flight: body xy accel vs onboard body velocity."""
    v = onboard_velocity(f)
    m = f['on'] & (f['t'] > start)
    vb = np.einsum('nji,nj->ni', f['R'][m], v[m])[:, :2]       # R^T v
    a = f['acc'][m][:, :2]
    a = a - a.mean(0)
    kd = -float(np.sum(a * vb) / np.sum(vb * vb))
    sigma = float(np.std(a + kd * vb))
    return kd, sigma


# ── replay ──────────────────────────────────────────────────────────────────
def replay(f, cfg, aid=True, snapshot_times=(), nll_from=None):
    """Aided replay. Returns (NLL of position innovations while flying after
    nll_from, {snapshot time: (kf copy, IMU index)})."""
    t, acc, R = f['t'], f['acc'], f['R']
    calib = t < t[0] + 1.5
    kf = k.InertialKF(cfg)
    kf.reset(f['p_ref'][0], k.calibrate_bias(acc[calib], R[calib]), t[np.nonzero(calib)[0][-1]])
    j_ref = int(np.searchsorted(f['t_ref'], kf.t))
    snaps = sorted(snapshot_times)
    out, nll, n = {}, 0.0, 0
    H = np.zeros((3, k.NX))
    H[:, k.P] = np.eye(3)
    Rm = np.eye(3) * REF_SIGMA ** 2
    for i in range(np.nonzero(calib)[0][-1] + 1, len(t)):
        kf.predict(acc[i], R[i], t[i])
        if not f['on'][i] and i % 5 == 0:
            kf.update_zero_velocity()
        while j_ref < len(f['t_ref']) and f['t_ref'][j_ref] <= t[i]:
            if aid:
                nu = f['p_ref'][j_ref] - kf.x[k.P]
                S = kf.P[k.P, k.P] + Rm
                if f['on'][i] and nll_from is not None and t[i] > nll_from:
                    nll += 0.5 * (nu @ np.linalg.solve(S, nu) + np.linalg.slogdet(S)[1])
                    n += 1
                kf._update(f['p_ref'][j_ref], H, Rm, gate=False)
            j_ref += 1
        while snaps and t[i] >= snaps[0]:
            out[snaps.pop(0)] = (copy.deepcopy(kf), i)
    return (nll / max(n, 1)), out


def coast(f, kf, i0, T, drag=False):
    """IMU only from index i0 for T s, as CrazyflieKF does after the handover.
    Returns (KF displacement, final 1-sigma)."""
    kf = copy.deepcopy(kf)
    p0 = kf.x[k.P].copy()
    t, acc, R = f['t'], f['acc'], f['R']
    i, buf = i0 + 1, []
    while i < len(t) and t[i] <= t[i0] + T:
        kf.predict(acc[i], R[i], t[i])
        if drag and kf.cfg.drag_k > 0:
            buf.append(acc[i])
            if len(buf) >= kf.cfg.drag_every:
                kf.update_drag(np.mean(buf, axis=0), R[i])
                buf = []
        i += 1
    return kf.x[k.P] - p0, kf.uncertainty()


def coast_test(f, cfg, Ts=(2.0, 4.0), drag=False, label=""):
    t_on = f['t'][f['on']]
    t_lo, t_hi = t_on[0] + 4.0, t_on[-1] - max(Ts) - 2.0
    starts = list(np.arange(t_lo, t_hi, 0.5))
    if not starts:
        print("flight too short for the coasting test")
        return
    _, snaps = replay(f, cfg, snapshot_times=starts)
    p_on = lambda tt: np.array([np.interp(tt, f['t_ref'], f['p_ref'][:, j]) for j in range(3)])
    for T in Ts:
        errs, inside = [], []
        for t0, (kf, i0) in snaps.items():
            d_kf, sig = coast(f, kf, i0, T, drag)
            d_on = p_on(t0 + T) - p_on(t0)
            e = d_kf - d_on
            errs.append(e)
            inside.append(np.all(np.abs(e) <= 3 * sig))
        e = np.array(errs)
        exy = np.linalg.norm(e[:, :2], axis=1)
        print(f"  {label:22s} {T:.0f} s coast, n={len(e):3d}: |xy| median {np.median(exy) * 100:5.1f} cm, "
              f"95% {np.percentile(exy, 95) * 100:5.1f} cm; |z| median "
              f"{np.median(np.abs(e[:, 2])) * 100:4.1f} cm; inside 3-sigma {np.mean(inside) * 100:3.0f}%")


def fit(f, cfg0, iters=150):
    from scipy.optimize import minimize
    names = ['acc_vrw', 'gm_sigma', 'gm_tau', 'tilt_sigma', 'tilt_tau']
    t_on = f['t'][f['on']]
    nll_from = t_on[0] + 3.0

    def obj(lx):
        c = replace(cfg0, **{n: float(math.exp(v)) for n, v in zip(names, lx)})
        try:
            return replay(f, c, nll_from=nll_from)[0]
        except np.linalg.LinAlgError:
            return 1e9

    x0 = np.log([max(getattr(cfg0, n), 1e-3) for n in names])
    print(f"[fit] start NLL {obj(x0):.3f}")
    res = minimize(obj, x0, method='Nelder-Mead',
                   options=dict(maxiter=iters, xatol=0.05, fatol=1e-3))
    best = {n: float(math.exp(v)) for n, v in zip(names, res.x)}
    print(f"[fit] NLL {res.fun:.3f} after {res.nfev} replays")
    for n in names:
        print(f"  {n:11s} {getattr(cfg0, n):10.4g} -> {best[n]:10.4g}")
    return replace(cfg0, **best)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('flight')
    ap.add_argument('--noise', help='starting noise model (imu_noise.json)')
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--coast', action='store_true')
    ap.add_argument('--drag', action='store_true')
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--out', default='imu_noise_flight.json')
    args = ap.parse_args()

    f = load(args.flight)
    cfg0 = k.KFConfig.from_json(args.noise) if args.noise else k.KFConfig()
    print(f"[replay] {args.flight}: {len(f['t'])} IMU samples, "
          f"{np.ptp(f['t'][f['on']]):.1f} s with motors on")

    cfg = cfg0
    if args.fit:
        cfg = fit(f, cfg0, args.iters)

    if args.drag:
        kd, sigma = fit_drag(f, f['t'][f['on']][0] + 3.0)
        cfg = replace(cfg, drag_k=kd, drag_sigma=sigma * DRAG_SIGMA_MULT)
        print(f"[drag] k = {kd:.3f} 1/s, residual sigma {sigma:.3f} m/s^2 "
              f"-> drag_sigma {cfg.drag_sigma:.3f} at {100 // cfg.drag_every} Hz")

    if args.fit or args.drag:
        with open(args.out, 'w') as fh:
            json.dump({"kf": {n: getattr(cfg, n) for n in
                              ('acc_vrw', 'gm_sigma', 'gm_tau', 'bias_rw', 'tilt_sigma',
                               'tilt_tau', 'drag_k', 'drag_sigma', 'drag_every')},
                       "source": f"kf_replay.py on {args.flight}"}, fh, indent=2)
        print(f"[replay] written to {args.out}")

    if args.coast:
        print("[coast] IMU-only from points in the aided flight, vs onboard displacement:")
        name = "fitted" if args.fit else "given"
        if args.fit:
            coast_test(f, cfg0, label="Allan (motors off)")
        coast_test(f, cfg, label=name)
        if cfg.drag_k > 0:
            coast_test(f, cfg, drag=True, label=name + " + drag")
