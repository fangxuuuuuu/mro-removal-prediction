"""Shared configuration: component type specs, sensor definitions, physical limits.

Single source of truth used by the data generator, feature builder and
preparation step. In a real MRO setting this would come from the
engineering data dictionary.
"""

TOTAL_CYCLES = 6000          # simulated fleet timeline length (flight cycles)
SNAPSHOT_EVERY = 20          # component checkpoints every N flight cycles
LABEL_WINDOW = 30            # prediction horizon: unscheduled removal within next N cycles
N_AIRCRAFT = 40
SLOTS_PER_AIRCRAFT = 7       # monitored component slots per aircraft
CYCLES_PER_DAY = 1.5         # utilisation assumption for day-based features
MAINT_INTERVAL = 1200        # scheduled aircraft maintenance interval (cycles)
SEED = 42

# Sensor column names (raw readings carried on every snapshot)
SENSOR_COLS = ["sensor_vibration", "sensor_oil_temp", "sensor_pressure_delta"]

# component_type -> spec
#   life        : mean flight cycles between removals for a nominal part
#   ata         : ATA chapter
#   sensor spec : (baseline value, slope per unit degradation, noise sigma)
TYPE_SPECS = {
    "APU":                dict(life=2400, ata="49", vib=(0.30, 1.4, 0.05), temp=(420.0, 60.0, 6.0),  press=(0.20, 1.6, 0.06)),
    "Engine_Fan":         dict(life=2800, ata="71", vib=(0.40, 2.2, 0.06), temp=(480.0, 55.0, 7.0),  press=(0.15, 1.2, 0.05)),
    "Hydraulic_Pump":     dict(life=1900, ata="29", vib=(0.22, 1.8, 0.04), temp=(65.0,  22.0, 3.0),  press=(1.80, 2.5, 0.12)),
    "Landing_Gear_Brake": dict(life=1500, ata="32", vib=(0.15, 1.1, 0.05), temp=(90.0,  40.0, 5.0),  press=(2.50, 1.8, 0.15)),
    "Fuel_Pump":          dict(life=2100, ata="28", vib=(0.25, 1.5, 0.04), temp=(40.0,  15.0, 2.5),  press=(3.20, 2.2, 0.14)),
    "Starter_Generator":  dict(life=1700, ata="80", vib=(0.20, 1.6, 0.05), temp=(70.0,  30.0, 3.5),  press=(0.80, 1.4, 0.08)),
}

EXPECTED_LIFE = {name: spec["life"] for name, spec in TYPE_SPECS.items()}

# Physical limits per sensor: values outside are treated as sensor faults and
# clipped (kept, not dropped - extreme readings are often the signal).
SENSOR_LIMITS = {
    "sensor_vibration":      (0.0, 6.0),
    "sensor_oil_temp":       (10.0, 900.0),
    "sensor_pressure_delta": (0.0, 12.0),
}
