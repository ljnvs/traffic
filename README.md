# Forecast-to-decision value in traffic control

This repository contains the core code and compact results for a study of how traffic-forecast quality relates to signal-control outcomes. The study uses reconstructed demand from Xuancheng, China, and compares causal forecasts with a matched no-future controller and a five-second event oracle.

The manuscript and publication figures are not hosted here. Raw and derived vehicle records, model archives, per-day flow files, and development environments are also omitted.

## Main result

On seven frozen Xuancheng test dates, the causal forecast changes queue vehicle-seconds by **-6.09%** and the five-second event oracle by **-3.39%** relative to the matched no-future rollout. Positive improvement means lower queues. The event oracle improves spillback exposure by **3.78%**. The corresponding date-paired bootstrap intervals all cross zero; these effects are exploratory.

## Repository map

- [`code/`](code): point and probabilistic forecasting code, the corrected control interface, final rollout, and focused tests.
- [`results/`](results): compact prediction, calibration, scenario, control, and uncertainty summaries.
- [`config/SPLIT.json`](config/SPLIT.json): frozen chronological date split.
- [`requirements.txt`](requirements.txt): Python package requirements.

## Reproduction scope

The published dataset must be obtained separately before retraining or rerunning closed-loop simulations. The extracted road network and per-day flows are omitted while their redistribution terms are checked. The controls use source-departure times as proxies for boundary arrivals and an aggregate store-and-forward queue model; the results are not microscopic or field-deployment measurements.

The final controller's entry point is [`code/run_control_corrected_rollout_v8.py`](code/run_control_corrected_rollout_v8.py). Its local project defaults expect the layout of the full research workspace; use the CLI path arguments when running against a separately prepared dataset. The date split in `config/` documents the frozen setup but does not substitute for the omitted network, per-day flow, and demand inputs.

Python 3.12 with NumPy, pandas, SciPy, scikit-learn, LightGBM, and Matplotlib was used. The focused control tests are in `code/test_control_*_v*.py` and require `pytest`.

The research data source is Ma et al., *City-scale high-resolution traffic datasets with refined networks for hierarchical traffic control*, *Scientific Data* (2026), DOI: [10.1038/s41597-026-06892-2](https://doi.org/10.1038/s41597-026-06892-2).
