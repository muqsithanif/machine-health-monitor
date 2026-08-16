# Machine Health Monitor

Detects mechanical faults from sensor data while they are still developing, and compares three ways of doing it on the measure a maintenance planner actually cares about.

![Bearing wear on pump-01](results/pump_01.png)

*Kurtosis, in the middle panel, starts climbing at the moment the fault begins while vibration energy in the top panel is still flat. Both are computed because of that gap.*

---

## Why lead time, and not accuracy

Accuracy is the obvious way to score a detector, and on this problem it is close to meaningless.

Consider a machine that runs healthy for two thirds of its life and then fails. A detector that never fires at all is correct on every healthy sample, which is two thirds of the data, so it scores about 65 % accuracy while providing nothing. Precision and recall improve on this, and they still miss the point, because they treat a warning as either right or wrong without asking when it arrived.

The operational question is different. A detector that flags a failing bearing three hours before it seizes is correct, and no maintenance planner can use it. Three hours is not enough time to schedule a crew, order a part, or arrange downtime. A detector that warns four days ahead, even at the cost of a few more false alarms, is the better instrument, and no accuracy-style metric will ever say so.

So every result below reports **lead time** first, meaning how many hours of warning the detector gave before the fault began, with **false alarms per week** beside it. That pair is the trade a planner has to make, and putting them together forces the comparison to be honest.

---

## What the study measures

Five simulated machines, thirty days each, sampled every ten minutes. Four develop a fault at a known moment; one stays healthy throughout as a control.

Each detector is fitted on the first 35 % of a run, which stands in for the commissioning period after installation. The alarm threshold is drawn from that window alone and never from the part of the run containing the failure. Choosing the threshold with knowledge of the failure would be tuning on the answer, and it is the most common way this kind of study flatters itself.

### The faults

Three modes, each following how the physical failure actually behaves.

**Bearing wear.** Kurtosis climbs first and vibration energy follows later with a knee. Early damage produces sharp isolated impacts, which change the shape of the signal well before they change its total energy.

**Imbalance.** Vibration energy grows with the square of severity while kurtosis barely moves, because the vibration stays smooth and repetitive rather than becoming impulsive.

**Progressive overheating.** Temperature drifts upward and motor current rises with it, because a bearing that is tightening demands more torque.

Each run also carries a handful of isolated spikes that are **not** faults: a load step, a passing forklift, a sensor glitch. They are there deliberately. A detector that flags them is producing false alarms, and a study that leaves them out flatters every method by the same amount and therefore compares nothing.

The data is simulated rather than taken from a public run-to-failure dataset, and that is a considered choice. Lead time can only be measured against a known fault onset, and in real datasets the moment a fault began is usually disputed.

---

## The three detectors

**Rolling z-score** compares each reading against the mean and spread of a trailing window, and reports how many standard deviations away the worst feature sits. It is the simplest thing that could work.

**EWMA control chart** keeps a running average that weights recent readings more heavily than old ones, and raises an alarm when that average drifts away from where it sat during commissioning. It was designed in the 1950s for exactly one purpose: finding a small persistent shift buried in noisy data, while ignoring one-off disturbances.

**Isolation Forest** is a machine-learning method that scores how easy a sample is to separate from the rest of the data. It looks for points that are individually unusual.

---

## Results

| Detector | Faults found | Mean lead time | Precision | Recall | False alarms / week | Transients flagged |
|---|---:|---:|---:|---:|---:|---:|
| **EWMA control chart** | **4/4** | **+226 h** | 0.79 | 0.57 | **5.9** | **0** |
| Rolling z-score | 4/4 | −87 h | 0.76 | 0.50 | 13.3 | 22 |
| Isolation Forest | 3/4 | −90 h | 0.55 | 0.02 | 10.8 | 16 |

Positive lead time means the alarm arrived before the fault began. Negative means it fired that many hours into the failure, when the machine was already degrading.

EWMA gives about nine days of warning. The other two respond only once degradation is well under way, by which point the vibration is obvious without a monitor at all.

### Why the simplest method wins

The result looks surprising until you line the method up against the problem.

A developing bearing fault is a small persistent shift buried in noise, alongside occasional disturbances that must be ignored. That is the exact shape EWMA was built for, so it wins for the reason it was designed.

Isolation Forest fails here for a reason that is just as specific. It searches for individually unusual points, and a slow drift never produces one. Every single sample during degradation falls inside a plausible range, and only the trend across them is abnormal. Its recall of 0.02 follows directly: it is not detecting the fault at all, it is firing on the transients, which genuinely are unusual and genuinely are not faults.

On the healthy control machine, Isolation Forest raised 30 false alarms per week. An operator would mute that monitor inside a fortnight, and after that it protects nothing.

---

## The features, and why there are three of them

Vibration analysts have used RMS, kurtosis and crest factor on rotating machinery for decades. They are kept together here because each one is blind to something the others catch.

**RMS** measures total vibration energy, and misses a sharp brief impact because a single spike barely moves an average over the whole window.

**Kurtosis** measures how impulsive the signal is, so it catches those impacts. It has an awkward property that matters in practice: late in a failure it falls back toward normal, once the damage is widespread enough that the signal looks like broad noise again. A monitor watching kurtosis alone can report that a badly worn bearing has recovered.

**Crest factor** is peak divided by RMS, and shares that weakness for the same reason.

### Correcting for load, and one attempt that failed

A machine vibrates more when it is working harder. Without correcting for that, every busy shift looks like a developing fault.

The obvious correction is to divide vibration by load, and it is wrong here. The relationship between them is affine rather than proportional, meaning there is a standing level of vibration even at low load. Dividing by a small load number amplifies that standing level instead of cancelling it, and the error is worst exactly when the machine is idling.

Regressing the load out with a trailing least-squares fit is the correct operation. It brings healthy-machine variation down to the sensor noise floor, and a test encodes the reasoning so that a later reader does not simplify it back into a ratio.

Having built the better feature, it turned out not to be automatically better as an input. Feeding the load residual to the EWMA chart instead of the plain rolling mean buys 44 more hours of lead time and starts flagging transients, eight across the fleet against zero before. The rolling mean smooths those disturbances away, and the residual by construction preserves them. The default keeps the quieter option, and the noisier one is one argument away.

---

## A condition this result depends on

The healthy machines in this study sit at a steady level with occasional transients. That turns out to be doing more work than it first appears.

The same three detectors were later run against a simulated batch reactor from [reactor-plc-trainer](https://github.com/muqsithanif/reactor-plc-trainer), a separate project modelling equipment fouling rather than bearing wear. On the reactor's healthy control machine, serviced on a fixed schedule so it never actually degrades, EWMA raised false alarms on 37 % of batches while Isolation Forest raised them on 0.8 %. The ranking reversed.

The reason follows from the same mechanism that makes EWMA win here. Servicing on a fixed interval makes wear climb and then reset, over and over. That sawtooth is a small persistent shift repeated, which is precisely what the chart was built to find, so it reports the maintenance schedule as a fault.

So the claim that EWMA wins because slow drift produces no individually unusual sample still holds, and it is not the whole story. The missing condition is that **the healthy baseline has to be stationary**. Where normal operation contains its own periodic structure, EWMA reads that structure as the fault it was looking for. The full comparison is in [that project's experiments folder](https://github.com/muqsithanif/reactor-plc-trainer/blob/main/experiments/README.md).

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

Thirty-nine tests. The ones worth naming exist because something specific went wrong.

- **A single sample over the threshold is not an alarm, six consecutive samples are.** Without that rule, every transient registers as a detection.
- **A warm-up period with no data is neither a flag nor a clean bill of health.** Filling it with zeros records unscored time as confirmed-healthy; filling it forward invents an alarm on the first sample.
- **The threshold cannot see the failure.** Selecting it from the whole run is tuning on the answer.
- **Features cannot see the future.** Truncating a run must leave every feature computed before the cut unchanged.
- **A ratio would not have worked.** The affine-versus-proportional property above, encoded so it is not simplified away later.
- **A spike decays while a shift persists.** Duration rather than peak height is the EWMA property that separates them.

That second test exists because the first version of this study reported eight days of lead time it had not earned. Missing values during warm-up were filled backward with a number above the fitted mean, the chart alarmed on the very first sample, and the evaluator scored the entire pre-fault period as early detection. The figure looked plausible, which is exactly what made it worth guarding against.

---

## Limits

**The data is simulated.** Degradation curves are modelled on published fault behaviour rather than measured from instrumented machines. A ranking computed on real run-to-failure data would differ in magnitude, though the mechanism behind the EWMA result does not depend on the simulation.

**One fault at a time.** Real machines fail in combination, and a second fault developing during the first is outside what this covers.

**Detection without diagnosis.** The system reports that something has changed. It does not identify which component to replace. Separating bearing wear from imbalance needs spectral analysis at a sampling rate far above the ten-minute interval used here.

---

Built by [Muqsit Muhammad Hanif](https://github.com/muqsithanif) · muqsithanif29@gmail.com
