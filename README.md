# Machine Health Monitor

Detects developing mechanical faults in sensor time series. Three detection methods are compared on lead time: the number of hours of warning each gives before a fault begins.

An EWMA control chart, a method published in 1959 and built on Shewhart's 1930s work, outperforms Isolation Forest here by a wide margin. Section [Why the simplest method wins](#why-the-simplest-method-wins) explains the mechanism.

![Bearing wear on pump-01](results/pump_01.png)

*Kurtosis (middle) begins climbing at fault onset while vibration RMS (top) remains flat. Both are computed because of that gap.*

---

## Why lead time, and not accuracy

In a run that stays healthy for two thirds of its length, a detector that never fires scores about 65% accuracy while providing no value. Precision and recall improve on this, and still miss the operational question.

A detector that flags a bearing three hours before seizure is correct and of no practical use. No maintenance planner can schedule a crew or order a part in three hours. A detector that warns four days ahead at the cost of a few more false alarms is the better instrument, and accuracy-style metrics do not capture that difference.

Every detector below reports lead time first, with false alarms per week beside it. That pair is the trade a maintenance planner has to make.

---

## Results

Five simulated machines, 30 days each, sampled at 10-minute intervals. Each detector is fitted on the first 35% of a run, taken as the commissioning period. The alarm threshold comes from that window alone and never from the portion of the run containing the failure.

| Detector | Faults found | Mean lead time | Precision | Recall | False alarms / week | Transients flagged |
|---|---|---:|---:|---:|---:|---:|
| **EWMA control chart** | **4/4** | **+226 h** | 0.79 | 0.57 | **5.9** | **0** |
| Rolling z-score | 4/4 | −87 h | 0.76 | 0.50 | 13.3 | 22 |
| Isolation Forest | 3/4 | −90 h | 0.55 | 0.02 | 10.8 | 16 |

Positive lead time means the alarm arrived before the fault began. Negative means it fired that many hours into the failure.

EWMA gives about nine days of warning. The other two methods respond once degradation is under way, at which point the vibration signature is visible without a monitor.

### Why the simplest method wins

EWMA was designed for this shape of problem: a small persistent shift buried in noise, alongside one-off disturbances that must be ignored. A developing bearing fault fits that description.

Isolation Forest searches for individually unusual points, and a slow drift produces none. Each sample during degradation falls within a plausible range, and only the trend is abnormal. Its recall of 0.02 follows from this. The method fires on the transients, which are unusual and are not faults.

On the healthy control machine, Isolation Forest raised 30 false alarms per week. An operator would mute that monitor within a fortnight, after which it protects nothing.

---

## What is being detected

Three failure modes, each following the degradation behaviour of the physical fault:

| Mode | Signature |
|---|---|
| **Bearing wear** | Kurtosis climbs first, RMS follows with a knee. Early spalling is impulsive well before it raises total energy |
| **Imbalance** | RMS grows with the square of severity while kurtosis moves little, since the vibration remains close to sinusoidal |
| **Progressive overheating** | Temperature drifts upward and current rises with it, as a tightening bearing demands more torque |

Each run also carries a small number of isolated spikes that are not faults: a load step, a passing forklift, a sensor glitch. A detector that flags these is producing false alarms. A study that omits them flatters every method by the same amount.

The data is simulated rather than drawn from a public run-to-failure dataset. Lead time can only be measured against a known fault onset, and in real datasets the onset is usually disputed.

---

## Features

RMS, kurtosis, and crest factor have been standard in rotating-machinery condition monitoring for decades. Each one is blind to something the others catch:

- **RMS** tracks total energy and misses a sharp, brief impact.
- **Kurtosis** tracks impulsiveness. It returns toward normal late in a failure, once the damage is widespread enough to resemble broad noise. A monitor reading kurtosis alone can report that a badly worn bearing has recovered.
- **Crest factor** shares that weakness, for the same reason.

### The load problem

A machine vibrates more under higher load. Without a correction for load, every busy shift resembles a developing fault.

A ratio of `vibration / load` is the obvious correction and it fails here. Vibration against load is affine rather than proportional, since a standing vibration level exists even at low load. Dividing by load amplifies the swing instead of cancelling it, and the error is largest when the machine idles. Regressing the load out with a trailing least-squares fit is the correct operation, and it brings healthy-machine variation down to the sensor noise floor.

The corrected feature does not automatically improve detection. Feeding the load residual to the EWMA chart in place of the plain rolling mean buys 44 additional hours of lead time and begins flagging transients: 8 across the fleet, against 0 for the plain mean. The rolling mean smooths those disturbances away and the residual, by construction, preserves them. The default keeps the quieter option, and `EwmaControlChart(column=...)` selects the other.

---

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_study.py
```

Writes per-machine plots, `per_machine.csv`, and `leaderboard.csv` into `results/`.

---

## Tests

```bash
pytest -q
```

39 tests. The ones worth naming:

- **A single sample over the threshold is not an alarm.** Six consecutive samples are. Without this rule, every transient registers as a detection
- **A NaN warm-up period is neither a flag nor a clean bill of health.** Filling it with zero records unscored time as confirmed-healthy; filling it forward invents an alarm on the first sample
- **The threshold cannot see the failure.** Selecting it from the whole run amounts to tuning on the answer
- **Features cannot see the future.** Truncating a run must leave every feature computed before the cut unchanged
- **A ratio would not have worked.** The affine-versus-proportional property above, encoded so that a later reader does not "simplify" it back
- **A spike decays while a shift persists.** Duration, not peak height, is the EWMA property that separates them

The warm-up test exists because the first version of this study reported eight days of lead time it had not earned. Backfilled NaN values sat above the fitted mean, the chart alarmed on the first sample, and the evaluator scored the entire pre-fault period as early detection. The resulting figure was plausible enough to pass review, which is what made it worth guarding against.

---

## Limits

**The data is simulated.** Degradation curves are modelled on published fault behaviour rather than measured from instrumented machines. A ranking computed on real run-to-failure data would differ in magnitude, though the mechanism behind the EWMA result does not depend on the simulation.

**The healthy baseline has to be stationary, and that condition is doing real work.** EWMA wins above because healthy machines here sit at a level with occasional transients. Tested against a batch reactor serviced on a fixed interval, where wear climbs and resets between services, EWMA read that sawtooth as the fault it was looking for and raised false alarms on 37% of batches. Isolation Forest, which loses badly in this study, raised them on 0.8% of the same data. Written up in [reactor-plc-trainer/experiments](https://github.com/muqsithanif/reactor-plc-trainer/blob/main/experiments/README.md).

**One fault at a time.** Real machines fail in combination, and a second fault developing during the first is outside the scope of this study.

**Detection without diagnosis.** The system reports that something has changed and does not identify the component to replace. Separating bearing wear from imbalance requires spectral analysis at a sampling rate far above the ten-minute interval used here.

---

Built by [Muqsit Muhammad Hanif](https://github.com/muqsithanif) · muqsithanif29@gmail.com
