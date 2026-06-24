Influence matrix overview
=========================

Input file:
D:\Sharon\MZA_Thesis\sa_plotting_scripts\sa_results\run_level_kpis.csv

Rows:
- Zoning variants.

Columns:
- Weather: Munich vs Napoli = median(KPI | TRY_B) - median(KPI | TRY_A), relative to Napoli.
- Retrofit: retrofit vs standard = median(KPI | retrofit) - median(KPI | standard), relative to standard.
- Continuous parameters: high quartile vs low quartile within each variant.
- P95-P05 spread columns: uncertainty range relative to P50.
- Seed variability: median coefficient of variation across repeated seeds for same variant/weather/sample.
- Variant vs 1Z / 11Z/A: median variant contrast relative to the selected baseline variant.

Important:
This is an overview/diagnostic matrix, not a causal decomposition. Continuous parameter columns are high-low contrasts from the run-level sample and can include correlations with other sampled inputs.

KPIs included:
- annual_heating_kWh: Annual heating demand
- peak_heating_kW: Peak heating load
- overheating_hours_any_zone_gt_26C: Overheating hours, any zone > 26 °C
- mean_tair_C: Mean air temperature
- max_tair_C: Maximum air temperature
- mean_interzone_spread_C: Mean interzonal spread
- max_interzone_spread_C: Maximum interzonal spread
