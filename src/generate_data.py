"""Synthetic mock dataset generator for unscheduled component removal prediction.

Simulates a fleet of aircraft where each monitored component slot goes through
a sequence of part instances. Each part degrades stochastically; when the
degradation crosses a threshold the part suffers an *unscheduled* removal
(the positive class), otherwise it is removed on schedule (negative).

Design goals:
- ~2% positive rate among snapshots (tunable via NOMINAL_LIFE_MULT).
- Realistic precursor signal: an incipient-failure window during which sensor
  trends rise before an unscheduled removal.
- Imperfect signal: sensor noise, missing readings, nuisance shocks, scheduled
  removals that look healthy, harsh vs. gentle aircraft.
- Strictly forward-looking labels: label=1 at snapshot cycle c iff the part
  suffers an unscheduled removal in (c, c + LABEL_WINDOW].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CYCLES_PER_DAY,
    LABEL_WINDOW,
    MAINT_INTERVAL,
    N_AIRCRAFT,
    SEED,
    SENSOR_COLS,
    SLOTS_PER_AIRCRAFT,
    SNAPSHOT_EVERY,
    TOTAL_CYCLES,
    TYPE_SPECS,
)

# Multiplier that controls how long parts last relative to TYPE_SPECS life.
# Smaller -> more removal events -> higher positive rate.
NOMINAL_LIFE_MULT = 0.9

TYPE_NAMES = sorted(TYPE_SPECS)
TYPE_PROBS = [0.20, 0.18, 0.18, 0.16, 0.14, 0.14]  # sum = 1.0


def _simulate_part_instance(
    rng: np.random.Generator,
    part_id: str,
    aircraft_id: str,
    comp_type: str,
    install_cycle: int,
    install_age: int,
    harsh: bool,
    aircraft_cycles: list[int],
    slot_repairs: list[int],
) -> tuple[pd.DataFrame, int]:
    """Simulate one physical part from installation to removal.

    Returns (snapshots DataFrame, removal cycle).
    """
    spec = TYPE_SPECS[comp_type]
    life = spec["life"]

    quality = rng.uniform(0.75, 1.30)
    if harsh:
        quality *= 0.65

    # Unscheduled removal cycle: exponential hazard around the nominal life.
    mean_life = max(life * quality * NOMINAL_LIFE_MULT, 200.0)
    r_unsched = install_cycle + rng.exponential(mean_life)
    # Scheduled removal: hard-life limit, looks healthy (no precursor window).
    r_sched = install_cycle + life * rng.uniform(1.05, 1.40)

    unscheduled = r_unsched < r_sched
    removal_cycle = int(min(r_unsched, r_sched))
    # guarantee at least one checkpoint before removal
    removal_cycle = max(removal_cycle, install_cycle + SNAPSHOT_EVERY + 1)

    # Incipient degradation window before an unscheduled failure.
    incipient = rng.uniform(120, 400) if unscheduled else 0.0

    rows = []
    ou = 0.0  # smooth Ornstein-Uhlenbeck-ish noise state
    fault_events: list[int] = []   # checkpoint indices where any fault code fired
    hard_fault_idx: list[int] = [] # checkpoint indices with a hard fault

    checkpoint = 0
    sim_end = min(removal_cycle, TOTAL_CYCLES)  # never generate beyond the fleet timeline
    for c in range(install_cycle, sim_end):
        if c % SNAPSHOT_EVERY != 0:
            continue
        age_ratio = (install_age + c - install_cycle) / life

        # --- latent degradation (0..1) ---
        base = 0.10 + 0.25 * min(age_ratio, 1.5)
        wear = 0.15 * (c - install_cycle) / max(mean_life, 1.0)
        accel = 0.0
        if unscheduled and c > removal_cycle - incipient:
            x = (c - (removal_cycle - incipient)) / incipient
            accel = 0.65 * x * x
        ou = 0.96 * ou + rng.normal(0.0, 0.02)
        dnorm = float(np.clip(base + wear + accel + ou, 0.0, 1.0))

        # --- sensors ---
        vib_b, vib_s, vib_n = spec["vib"]
        tmp_b, tmp_s, tmp_n = spec["temp"]
        prs_b, prs_s, prs_n = spec["press"]
        raw = {
            "sensor_vibration": rng.normal(vib_b + vib_s * dnorm, vib_n),
            "sensor_oil_temp": rng.normal(tmp_b + tmp_s * dnorm, tmp_n),
            "sensor_pressure_delta": rng.normal(prs_b + prs_s * dnorm, prs_n),
        }

        # --- missingness: 8% per sensor + occasional correlated outage ---
        outage = rng.random() < 0.02
        for col in SENSOR_COLS:
            p_missing = 0.05 + (0.15 if outage else 0.0)
            if rng.random() < p_missing:
                raw[col] = np.nan

        # --- fault codes ---
        n_codes = rng.poisson(0.3 + 2.5 * dnorm)
        if n_codes > 0:
            fault_events.append(checkpoint)
        hard = rng.random() < (0.004 + 0.10 * dnorm)
        if hard:
            hard_fault_idx.append(checkpoint)

        last_90d = [i for i in fault_events if checkpoint - i <= max(1, int(135 / SNAPSHOT_EVERY))]
        last_30d = [i for i in hard_fault_idx if checkpoint - i <= max(1, int(45 / SNAPSHOT_EVERY))]

        last_maint = (c // MAINT_INTERVAL) * MAINT_INTERVAL
        days_since_maint = int((c - last_maint) / CYCLES_PER_DAY)
        repairs_180d = sum(
            1 for rc in slot_repairs if 0 <= c - rc <= int(180 * CYCLES_PER_DAY)
        )

        rows.append(
            dict(
                snapshot_id=f"{part_id}-C{c:06d}",
                aircraft_id=aircraft_id,
                component_id=part_id,
                component_type=comp_type,
                ata_chapter=spec["ata"],
                snapshot_cycle=c,
                cycles_since_install=install_age + c - install_cycle,
                sensor_vibration=raw["sensor_vibration"],
                sensor_oil_temp=raw["sensor_oil_temp"],
                sensor_pressure_delta=raw["sensor_pressure_delta"],
                fault_code_count_90d=len(last_90d),
                has_hard_fault=int(len(last_30d) > 0),
                days_since_last_maint=days_since_maint,
                repairs_last_180d=repairs_180d,
            )
        )
        checkpoint += 1

    # --- labels: unscheduled removal within the next LABEL_WINDOW cycles ---
    snapshots = pd.DataFrame(rows)
    if snapshots.empty:
        snapshots["label_removed_30d"] = pd.Series(dtype=int)
        return snapshots, removal_cycle
    if unscheduled:
        mask = (snapshots["snapshot_cycle"] < removal_cycle) & (
            snapshots["snapshot_cycle"] >= removal_cycle - LABEL_WINDOW
        )
        snapshots["label_removed_30d"] = mask.astype(int)
    else:
        snapshots["label_removed_30d"] = 0

    # register repair event for the slot (removal + swap)
    slot_repairs.append(removal_cycle)
    aircraft_cycles.append(removal_cycle)
    return snapshots, removal_cycle


def generate_dataset(
    n_aircraft: int = N_AIRCRAFT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate the full fleet dataset. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    frames = []
    part_counter = 0

    for a_idx in range(n_aircraft):
        aircraft_id = f"AC-{a_idx + 1:03d}"
        harsh = rng.random() < 0.30  # harsh-operating tails fail more often

        for slot in range(SLOTS_PER_AIRCRAFT):
            comp_type = rng.choice(TYPE_NAMES, p=TYPE_PROBS)
            aircraft_cycles: list[int] = []
            slot_repairs: list[int] = []
            cycle = 0
            # First part starts with random mid-life age for diversity.
            install_age = int(rng.uniform(0, TYPE_SPECS[comp_type]["life"] * 0.5))
            first = True
            while cycle < TOTAL_CYCLES:
                part_counter += 1
                part_id = f"CMP-{part_counter:05d}"
                snaps, removal = _simulate_part_instance(
                    rng, part_id, aircraft_id, comp_type,
                    cycle, install_age if first else 0,
                    harsh, aircraft_cycles, slot_repairs,
                )
                if not snaps.empty:
                    frames.append(snaps)
                first = False
                cycle = removal
                install_age = 0
                if removal >= TOTAL_CYCLES:
                    break

    df = pd.concat(frames, ignore_index=True)
    df["snapshot_cycle"] = df["snapshot_cycle"].astype(int)
    df["label_removed_30d"] = df["label_removed_30d"].astype(int)
    df = df.sort_values(["aircraft_id", "component_id", "snapshot_cycle"]).reset_index(drop=True)
    return df


def main(output_path: str = "data/component_snapshots.csv") -> None:
    df = generate_dataset()
    rate = df["label_removed_30d"].mean()
    print(f"rows={len(df):,}  components={df['component_id'].nunique():,}  "
          f"aircraft={df['aircraft_id'].nunique()}  positive_rate={rate:.3%}")
    print(df["label_removed_30d"].value_counts())
    df.to_csv(output_path, index=False)
    print(f"written -> {output_path}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/component_snapshots.csv"
    main(out)
