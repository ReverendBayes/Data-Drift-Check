## Drift-Check
# Tabular Data Drift Detector & Reporter

Tabular Data Drift Detector &amp; Reporter: A CLI tool that connects to any database or CSV, computes statistical drift (KS-test, Jensen-Shannon) between “baseline” vs. “current” data, and emits a Markdown/HTML report with charts.

## What it is 
A CLI tool that connects to any database or CSV, computes statistical drift (KS-test, Jensen-Shannon) between “baseline” vs. “current” data, and emits a Markdown/HTML report with charts.

## Why you need it
* MLOps gap: You need data-drift checks in production, but frameworks like Evidently or Great Expectations are heavyweight.
* Teams want a bullet-proof, “run-me-with-one-command” script for nightly checks.

## Key features
python drift-check.py \
  --baseline base.csv \
  --current new.csv \
  --report drift_report.md
* Auto-detects numeric vs. categorical
* Flags features with > 0.1 JS divergence
* Generates simple Matplotlib histograms inline


**Usage example**:

```bash
python driftcheck.py \
  --baseline base.csv \
  --current new.csv \
  --report drift_report.md
```

**Dependencies**:

```bash
pip install pandas numpy scipy matplotlib
```
