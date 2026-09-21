"""Read live training progress and test whether the score curve has flattened."""
import glob, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else "runs/6x6-qnorm"
ea = EventAccumulator(sorted(glob.glob(f"{run}/tb/*"))[0]); ea.Reload()
tags = ea.Tags()["scalars"]
get = lambda t: ea.Scalars(t) if t in tags else []

sc, steps, ent = get("train/mean_score"), get("train/mean_steps"), get("train/entropy")
ev = get("eval/mean_score")
n = len(sc)
elapsed = (sc[-1].wall_time - sc[0].wall_time) / 3600 if n > 1 else 0
per = (sc[-1].wall_time - sc[0].wall_time) / (n - 1) if n > 1 else 0

print(f"{run}: {n} iterations, {elapsed:.1f}h elapsed, {per:.0f}s/iter")
if n > 1:
    print(f"projected remaining to 200: {per*(200-n)/3600:.1f}h")
print()

print(f"{'iter':>5}{'selfplay':>10}{'steps':>8}{'entropy':>9}")
idx = list(range(0, n, max(1, n // 12))) + ([n - 1] if n else [])
for i in sorted(set(idx)):
    print(f"{sc[i].step:>5}{sc[i].value:>10.2f}{steps[i].value:>8.0f}{ent[i].value:>9.4f}")

if ev:
    print()
    print("eval (100 deterministic games, every 5 iterations):")
    for e in ev[-8:]:
        print(f"  iter {e.step:>4}  {e.value:6.2f}")

# Flatline test: compare the last quarter against the one before it.
if n >= 12:
    v = np.array([s.value for s in sc])
    q = max(3, n // 4)
    prev, last = v[-2*q:-q], v[-q:]
    slope = np.polyfit(np.arange(len(last)), last, 1)[0]
    print()
    print("FLATLINE CHECK")
    print(f"  previous {q} iters: mean {prev.mean():6.2f}")
    print(f"  latest   {q} iters: mean {last.mean():6.2f}   change {last.mean()-prev.mean():+.2f}")
    print(f"  slope over latest {q}: {slope:+.3f} per iteration")
    noise = last.std()
    print(f"  within-window noise (std): {noise:.2f}")
    flat = abs(last.mean() - prev.mean()) < noise and abs(slope) * q < noise
    print(f"  verdict: {'FLATLINING - improvement is below noise' if flat else 'still moving'}")
else:
    print()
    print(f"FLATLINE CHECK: need >=12 iterations, have {n}")
