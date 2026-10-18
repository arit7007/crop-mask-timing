import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Base paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
DATA_PATH_RAW = ROOT / "data" / "raw"

# API settings
NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

# Load credentials from .env
load_dotenv(ROOT / ".env")
NASS_API_KEY = os.getenv("NASS_API_KEY")
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")

# --- Crops & Geographic Scope ---
DEFAULT_STATES = [
    "IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID",
    "KS", "ND", "SD", "IN", "OH", "MO", "WI", "MS", "GA", "NC",
]

# Rainfed Corn Belt + Northern Plains modeling scope
STUDY_STATES = [
    "IA", "IL", "IN", "OH",  # Core Corn Belt
    "MN", "WI", "MO",        # Extended Corn Belt
    "KS", "ND", "SD", "NE",  # Northern Great Plains
]

# Note: CA is excluded due to sparse field crop coverage (mostly specialty crops).
STUDY_CROPS = [
    "CORN",
    "SOYBEANS",
    "WHEAT",
    "SORGHUM",
    "BARLEY",
    "OATS",
]

# Analysis window starts in 2008 because CDL crop masks begin there.
ANALYSIS_YEARS = (2008, 2025)

# --- Growing Seasons ---
DEFAULT_GROWING_SEASON = (4, 10)  # Apr-Sep

GROWING_SEASON_EXCEPTIONS = {
    "TX": (3, 9),
    "LA": (3, 9),
    "AR": (3, 10),
    "MN": (5, 10),  # Later start due to frost
}

def get_growing_season(state_abbr: str):
    """Get growing season tuple (start_month, end_month_exclusive) for a state."""
    return GROWING_SEASON_EXCEPTIONS.get(state_abbr.upper(), DEFAULT_GROWING_SEASON)


def get_required_months(slack: int = 1):
    """Derive needed download months from configured growing seasons plus slack."""
    seasons = [DEFAULT_GROWING_SEASON] + list(GROWING_SEASON_EXCEPTIONS.values())
    lo = min(s[0] for s in seasons) - slack
    hi = max(s[1] - 1 for s in seasons) + slack
    return list(range(max(1, lo), min(12, hi) + 1))


# --- CDL Codes for key crops ---
CDL_CODES = {
    "CORN": [1],
    "COTTON": [2],
    "RICE": [3],
    "SORGHUM": [4],
    "SOYBEANS": [5],
    "BARLEY": [21],
    "WHEAT": [22, 23, 24],
    "OATS": [28],
}

CROP_SPECIFIC_MASK_CROPS = ["CORN", "SOYBEANS"]


def get_cdl_codes(crop_name: str):
    """Retrieve CDL code list for a given crop name."""
    name_upper = crop_name.upper().strip()
    if name_upper not in CDL_CODES:
        available = ", ".join(CDL_CODES.keys())
        raise KeyError(f"Unknown crop: '{crop_name}'. Available: {available}")
    return CDL_CODES[name_upper]


# --- NASS Filters & Practice Modes ---
NASS_SKIP_COMMODITIES = {
    "HAY",
    "HAY & HAYLAGE",
}

NASS_SKIP_CLASS_DESC = {
    "WINTER",
}

# Mode options: "aggregate", "split", or "both"
NASS_PRACTICE_MODE = "aggregate"

NASS_PRACTICE_LABELS = {
    "ALL PRODUCTION PRACTICES": "all",
    "IRRIGATED": "irrigated",
    "NON-IRRIGATED": "non_irrigated",
}

NASS_PRACTICE_MODES = {
    "aggregate": ["ALL PRODUCTION PRACTICES"],
    "split": ["IRRIGATED", "NON-IRRIGATED"],
    "both": ["ALL PRODUCTION PRACTICES", "IRRIGATED", "NON-IRRIGATED"],
}

IRRIGATION_COLUMN = "irrigation"

# --- Helper Functions ---
def resolve_states(states=None, study: bool = False):
    """Resolve active state list based on study flag or custom inputs."""
    if study:
        return list(STUDY_STATES)
    return list(states) if states else list(DEFAULT_STATES)


def yield_filename(mode: str = None):
    """Determine output filename based on practice mode."""
    mode = (mode or NASS_PRACTICE_MODE).lower()
    return "yield.csv" if mode == "aggregate" else f"yield_{mode}.csv"


def get_practice_filter(mode: str = None):
    """Get raw NASS production practice strings to keep."""
    mode = (mode or NASS_PRACTICE_MODE).lower()
    if mode not in NASS_PRACTICE_MODES:
        raise KeyError(
            f"Unknown NASS_PRACTICE_MODE: {mode!r}. "
            f"Available: {', '.join(NASS_PRACTICE_MODES)}"
        )
    return NASS_PRACTICE_MODES[mode]