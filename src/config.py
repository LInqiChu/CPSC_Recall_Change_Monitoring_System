# src/config.py
# Configuration file for CPSC Recall Change Monitoring System

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Data paths
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
GOLD_DATA_DIR = os.path.join(DATA_DIR, "gold")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

# Gold standard file
GOLD_STANDARD_PATH = os.path.join(
    GOLD_DATA_DIR,
    "frozen_human_gold_current_window_v2.csv"
)

# Raw recall snapshots
RAW_SNAPSHOT_OLD = os.path.join(RAW_DATA_DIR, "cpsc_recalls_20260304.csv")
RAW_SNAPSHOT_NEW = os.path.join(RAW_DATA_DIR, "cpsc_recalls_20260430.csv")

# Output files
EVAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "auto_eval.csv")
SUMMARY_EVAL_PATH = os.path.join(OUTPUT_DIR, "summary_eval_sheet.csv")

# Demo path
DEMO_DASHBOARD_PATH = os.path.join(ROOT_DIR, "demos", "cpsc_recall_dashboard.html")