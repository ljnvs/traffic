# Forecast-to-decision value in traffic control

This is a compact research companion for the preprint *When Better Traffic Forecasts Fail to Improve Signal Control: A Layered Diagnostic Study of Forecast-to-Decision Value* by Jianing Long, Xiaobin Li, Wuming Lei, and Weiguang Wang.

The repository contains the paper, four publication figures, the code path for the corrected control audit and final experiment, and small result tables. It intentionally omits raw and derived vehicle records, model archives, flow files, development environments, and intermediate experiment versions.

## Main result

On seven frozen Xuancheng test dates, the causal forecast changes queue vehicle-seconds by **-6.09%** and the five-second event oracle by **-3.39%** relative to the matched no-future rollout. Positive improvement means lower queues. The event oracle improves spillback exposure by **3.78%**. Date-paired bootstrap intervals cross zero, so these are exploratory effects rather than general performance guarantees.

## Repository map

- [`paper/preprint.pdf`](paper/preprint.pdf): four-author arXiv-style preprint.
- [`paper/main.tex`](paper/main.tex), [`paper/references.bib`](paper/references.bib), and [`paper/arxiv.sty`](paper/arxiv.sty): editable manuscript source.
- [`paper/media/`](paper/media): four vector figures. `paper/make_figures.py` regenerates them from frozen numbers and the small v8 statistics table.
- [`code/`](code): core Python implementation and focused tests for the final causal forecast, event oracle, and joint-action rollout.
- [`results/`](results): compact prediction, calibration, scenario, control, and uncertainty summaries.
- [`config/SPLIT.json`](config/SPLIT.json): frozen chronological date split.

## Reproduction scope

The included code is the executable research path, but the published dataset must be obtained separately before retraining or rerunning closed-loop simulations. The controls use source-departure times as proxies for boundary arrivals and an aggregate store-and-forward queue model. They are not microscopic or field-deployment results. The extracted road network and per-day flows are omitted while redistribution terms are checked.

The final controller's entry point is `code/run_control_corrected_rollout_v8.py`. Its local project defaults expect the layout of the full research workspace; use the CLI path arguments when running against a separately prepared dataset. The date split in `config/` documents the frozen setup but does not substitute for the omitted network, per-day flow, and demand inputs.

Python 3.12 with NumPy, pandas, SciPy, scikit-learn, LightGBM, and Matplotlib was used. The four figures can be regenerated from the checked-in summary by running `python paper/make_figures.py` from this repository root.

The research data source is Ma et al., *City-scale high-resolution traffic datasets with refined networks for hierarchical traffic control*, *Scientific Data* (2026), DOI: [10.1038/s41597-026-06892-2](https://doi.org/10.1038/s41597-026-06892-2). See the paper for study limitations and references.
