from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# MAIN DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "01_data"

SCRIPTS_DIR = PROJECT_ROOT / "02_scripts"

RESULTS_DIR = PROJECT_ROOT / "03_results"

FIGURES_DIR = PROJECT_ROOT / "04_figures"

REPORT_DIR = PROJECT_ROOT / "05_report"

PRESENTATION_DIR = PROJECT_ROOT / "06_presentation"


# ============================================================
# RESULT SUBDIRECTORIES
# ============================================================

DIABETES_RESULTS_DIR = RESULTS_DIR / "diabetes"

HEART_RESULTS_DIR = RESULTS_DIR / "heart_disease"

STATISTICAL_RESULTS_DIR = RESULTS_DIR / "statistical_tests"


# ============================================================
# FIGURE SUBDIRECTORIES
# ============================================================

DIABETES_FIGURES_DIR = FIGURES_DIR / "diabetes"

HEART_FIGURES_DIR = FIGURES_DIR / "heart_disease"

CROSS_DATASET_FIGURES_DIR = FIGURES_DIR / "cross_dataset"


# ============================================================
# DATASETS
# ============================================================

DIABETES_ZIP = DATA_DIR / "archive.zip"

HEART_DISEASE_ZIP = DATA_DIR / "heart+disease.zip"