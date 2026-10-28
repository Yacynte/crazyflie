### crazyflie

Distributed consensus formation flight with Crazyflies (cflib).

| File | What it is |
|---|---|
| `drone.py` | `Drone` (real), `SimDrone`, `Message`, `Channel`, `Topology`, `ConsensusAgent`, `ConsensusSwarm` |
| `patterns.py` | formation patterns as functions of time: hover, line, circle, eight, swap |
| `run_swarm.py` | example: fly a pattern in sim or on hardware, optional plot |

```bash
python run_swarm.py --sim --pattern circle --plot circle.png
python run_swarm.py --sim --pattern eight --n 3 --topology chain --loss 0.1 --delay 0.05
python run_swarm.py --sim --pattern circle --n 3 --scenario leave_join
python run_swarm.py --pattern hover --duration 8        # hardware, edit HARDWARE in run_swarm.py
```

Each agent only sees its own state and the messages the `Topology` routes to it.
The PC runs the loop for now (distributed algorithm, centralized execution);
the agent logic is written so it can move onboard with the P2P API later.
