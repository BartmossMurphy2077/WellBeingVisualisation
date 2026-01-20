# WellBeingVisualisation

# Sources

- all the datasets were acquired from: http://github.com/open-numbers/ddf--gapminder--systema_globalis/tree/master

# Wellness Insights Analysis

This project includes a reproducible analysis package that builds a data audit and insights report
from the CSVs in `dataset/`.

## Quick Start

1) Install dependencies

```
python3 -m pip install -r requirements.txt
```

2) Run the analysis

```
python3 scripts/run_analysis.py --data-dir dataset --out-dir artifacts
```

Outputs are written to a timestamped run folder inside `artifacts/`.
