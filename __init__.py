"""
DQM Anomaly Detection Pipeline

A modular pipeline for CRV detector anomaly detection using multiple statistical
and machine learning approaches (Z-Score, PCA, TranAD).

Phases:
  1. Ingestion   - Load filtered CSV, validate timestamps, handle timezones
  2. Preprocessing - Feature selection, scaling, sliding window tensor creation
  3. Models      - Z-Score, PCA, and TranAD anomaly detection
  4. Run Pipeline - Orchestrate all phases and generate reports
"""

__version__ = "0.1.0"
__author__ = "Sophie"

from . import ingestion, preprocessing, models, diagnostics

__all__ = ["ingestion", "preprocessing", "models", "diagnostics"]
