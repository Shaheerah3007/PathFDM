"""Download the trajectory data of the paper from Zenodo (DOI 10.5281/zenodo.23975078) into data/.

    python scripts/download_data.py                      # all five systems (~16 GB)
    python scripts/download_data.py alanine_dipeptide oame_g2
    PATHFDM_DATA=/scratch/pathfdm python scripts/download_data.py trypsin_benzamidine

Every archive is checked against SHA256SUMS and unpacked into data/<system>/ (or $PATHFDM_DATA/<system>/).
The notebooks also download a missing system on first use, so this step is optional.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathfdm.data import SYSTEMS, fetch  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("systems", nargs="*", help=f"systems to download (default: all): {', '.join(SYSTEMS)}")
p.add_argument("--dest", help="data root (default: $PATHFDM_DATA or <repository>/data)")
p.add_argument("--keep-archives", action="store_true", help="keep the .tar files after unpacking")
p.add_argument("--force", action="store_true", help="download and unpack again even if the folder exists")
a = p.parse_args()
fetch(a.systems or None, dest=a.dest, keep_archives=a.keep_archives, force=a.force)
