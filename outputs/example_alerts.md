# Example Alert Cards (test window, highest-risk components)

### Alert 1 — CMP-01409 (Fuel_Pump, aircraft AC-038)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 5260 · age 1758 cycles · hard fault: yes
- **Top factors:** sensor_vibration_ratio_baseline=2.620 (contribution +3.28); sensor_vibration_std5=0.255 (contribution +1.33); sensor_oil_temp_ratio_baseline=1.293 (contribution +1.23); sensor_vibration_delta15=1.045 (contribution +0.53)

### Alert 2 — CMP-01022 (Hydraulic_Pump, aircraft AC-027)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 4820 · age 6 cycles · hard fault: no
- **Top factors:** sensor_vibration_ratio_baseline=1.985 (contribution +1.69); sensor_vibration_std5=n/a (contribution +1.53); cycles_since_install=6 (contribution +0.87); age_ratio=0.003 (contribution +0.80)

### Alert 3 — CMP-01069 (APU, aircraft AC-029)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 4900 · age 1118 cycles · hard fault: no
- **Top factors:** sensor_vibration_std5=0.255 (contribution +2.36); sensor_vibration_ratio_baseline=2.063 (contribution +1.58); sensor_pressure_delta_ratio_baseline=2.304 (contribution +0.69); sensor_pressure_delta_delta15=n/a (contribution +0.46)

### Alert 4 — CMP-00064 (Starter_Generator, aircraft AC-002)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 4960 · age 607 cycles · hard fault: no
- **Top factors:** sensor_vibration_ratio_baseline=2.438 (contribution +2.46); sensor_vibration_std5=0.254 (contribution +1.92); sensor_oil_temp_ratio_baseline=1.182 (contribution +0.70); sensor_vibration_delta15=n/a (contribution +0.50)

### Alert 5 — CMP-01118 (APU, aircraft AC-030)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 5800 · age 149 cycles · hard fault: no
- **Top factors:** sensor_vibration_std5=0.258 (contribution +2.10); sensor_vibration_ratio_baseline=1.838 (contribution +0.72); age_ratio=0.062 (contribution +0.61); cycles_since_install=149 (contribution +0.60)

### Alert 6 — CMP-00946 (Starter_Generator, aircraft AC-026)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 5300 · age 83 cycles · hard fault: no
- **Top factors:** sensor_vibration_ratio_baseline=2.170 (contribution +2.04); sensor_vibration_std5=0.233 (contribution +1.24); sensor_oil_temp_ratio_baseline=1.260 (contribution +1.06); cycles_since_install=83 (contribution +0.69)

### Alert 7 — CMP-01021 (Hydraulic_Pump, aircraft AC-027)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 4800 · age 341 cycles · hard fault: no
- **Top factors:** sensor_vibration_ratio_baseline=2.049 (contribution +1.80); sensor_vibration_std5=n/a (contribution +1.46); age_ratio=0.179 (contribution +0.77); cycles_since_install=341 (contribution +0.66)

### Alert 8 — CMP-00311 (APU, aircraft AC-009)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 5520 · age 2250 cycles · hard fault: yes
- **Top factors:** sensor_vibration_ratio_baseline=2.317 (contribution +2.35); sensor_pressure_delta_ratio_baseline=2.484 (contribution +1.01); age_ratio=0.938 (contribution -0.66); sensor_vibration_std5=0.200 (contribution +0.66)

### Alert 9 — CMP-00997 (APU, aircraft AC-027)
- **Risk score:** 100/100 — HIGH
- **Snapshot cycle:** 5100 · age 303 cycles · hard fault: yes
- **Top factors:** sensor_vibration_ratio_baseline=1.978 (contribution +1.35); sensor_pressure_delta_delta15=1.125 (contribution +0.70); sensor_vibration_delta15=1.042 (contribution +0.67); sensor_vibration_std5=0.211 (contribution +0.54)

### Alert 10 — CMP-00516 (Fuel_Pump, aircraft AC-015)
- **Risk score:** 99/100 — HIGH
- **Snapshot cycle:** 5680 · age 1355 cycles · hard fault: no
- **Top factors:** sensor_vibration_std5=0.297 (contribution +2.54); sensor_vibration_ratio_baseline=2.252 (contribution +2.02); sensor_oil_temp_ratio_baseline=1.205 (contribution +0.64); sensor_pressure_delta_delta15=1.423 (contribution +0.63)


# Global Feature Importance (XGBoost, gain)

| Rank | Feature | Gain |
|---|---|---|
| 1 | sensor_vibration_ratio_baseline | 795.0 |
| 2 | sensor_vibration_std5 | 395.4 |
| 3 | sensor_vibration_delta15 | 229.8 |
| 4 | sensor_oil_temp_ratio_baseline | 226.3 |
| 5 | sensor_pressure_delta_delta15 | 173.4 |
| 6 | sensor_pressure_delta_ratio_baseline | 164.2 |
| 7 | age_ratio | 53.0 |
| 8 | cycles_since_install | 40.3 |
| 9 | component_type | 31.9 |
| 10 | sensor_vibration | 29.8 |
| 11 | sensor_oil_temp_delta15 | 29.1 |
| 12 | sensor_pressure_delta_std5 | 27.0 |
| 13 | sensor_oil_temp | 26.8 |
| 14 | sensor_oil_temp_mean5 | 21.2 |
| 15 | sensor_pressure_delta | 20.8 |
