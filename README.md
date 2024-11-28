# Climate Stabilisation Under Net Zero CO2 Emissions
This repository contains the code used to analyze the stability of global and regional temperature responses under net-zero CO2 emissions. The analysis employs the Signal-to-Noise (S/N) ratio methodology, applied to simulations from the Zero Emissions Commitment Model Intercomparison Project (ZECMIP).

## Repository Contents and Structure

Code used to generate figures analyzing the global mean surface temperature (Figure 1 and Figures S1-S6) can be found in **`01_global_mean_stability.ipynb`**.

Code used to analyze the local temperature response (Figures 2-7 and Figures S7-S8) can be found in **`01_regional_stability.ipynb`**.

## Code for Replicating Stabilisation Algorithm

To replicate the stabilisation algorithm, the easiest starting point is **`01_global_stability_demonstration.ipynb`**.

The required functions can be found in **`src/signal_to_noise_calculations.py`**:

- **`signal_to_noise_ratio`**: Calculates the Signal-to-Noise (S/N) ratio for a single window.
- **`signal_to_noise_ratio_multi_window`**: Calculates the S/N ratio for multiple windows.
- **`calculate_year_stable_and_unstable`**: Identifies the years at which non-stabilisation and stabilisation occur.
