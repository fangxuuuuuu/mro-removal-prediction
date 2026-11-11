# Mock Dataset — Component Snapshots

Synthetic MRO dataset for the unscheduled-removal prediction PoC.
Generated deterministically by `src/generate_data.py` (seed 42).

`component_snapshots.csv` — 84,000 rows, one row = one component observation
(checkpoint every 20 flight cycles).

| Column | Type | Description |
|---|---|---|
| snapshot_id | string | Unique row key `{component_id}-C{cycle}` |
| aircraft_id | string | Tail number (group key, e.g. AC-001) |
| component_id | string | Physical part instance (group key, e.g. CMP-00042) |
| component_type | category | APU / Engine_Fan / Hydraulic_Pump / Landing_Gear_Brake / Fuel_Pump / Starter_Generator |
| ata_chapter | category | ATA chapter of the component type |
| snapshot_cycle | int | Flight cycle at snapshot |
| cycles_since_install | int | Age of the part instance |
| sensor_vibration | float | Vibration amplitude (~8% missing) |
| sensor_oil_temp | float | Oil temperature (~8% missing) |
| sensor_pressure_delta | float | Pressure drop across component (~8% missing) |
| fault_code_count_90d | int | Fault-code events in last 90 days |
| has_hard_fault | int | Hard fault in last 30 days |
| days_since_last_maint | int | Days since last scheduled maintenance |
| repairs_last_180d | int | Removal/repair events for this slot in last 180 days |
| label_removed_30d | int | **Target: 1 = unscheduled removal within next 30 flight cycles** (~1.76% positive) |

Notes:
- Labels are strictly forward-looking (never use future information as features).
- Sensors are noisy views of a latent degradation process; degradation accelerates
  120–400 cycles before unscheduled failures.
- 30% of aircraft operate in "harsh" conditions (higher failure rate).
- Scheduled removals occur at hard-life limits and carry label 0 (healthy-looking).
