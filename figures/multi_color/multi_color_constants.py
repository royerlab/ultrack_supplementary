from pathlib import Path

ROOT_DIR = Path("<DEFINE BY USER>")
CTC_RES_DIR = ROOT_DIR / "curated_tracks/RES"
CTC_GT_DIR = ROOT_DIR / "curated_tracks/TRA"

FIG_DIR = Path(".")
FIG_DIR.mkdir(exist_ok=True)
