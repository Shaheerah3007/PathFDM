"""Download the trajectory data of the paper from Zenodo into the data folder.

    from pathfdm.data import fetch
    fetch(["alanine_dipeptide", "oame_g2"])          # -> <repository>/data/alanine_dipeptide, ...

The archives (one tar per system) are checked against SHA256SUMS of the record and unpacked into the data root:
``$PATHFDM_DATA`` if set, else ``<repository>/data``. The loaders in systems/<system>/load_data.py call
``require()``, which downloads a missing system on first use.

While the record is an unpublished draft its files need a Zenodo token with the deposit scopes: set
``ZENODO_TOKEN`` or put the token in ``~/.zenodo_token``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

RECORD_ID = 23975078
DOI = "10.5281/zenodo.23975078"
API = "https://zenodo.org/api"
SYSTEMS = ["alanine_dipeptide", "oame_g2", "trypsin_benzamidine", "t4l_benzene", "imatinib_abl"]


def data_root() -> Path:
    return Path(os.environ.get("PATHFDM_DATA", Path(__file__).resolve().parents[1] / "data"))


def _token():
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok and (Path.home() / ".zenodo_token").exists():
        tok = (Path.home() / ".zenodo_token").read_text().strip()
    return tok or None


def _get(url, token=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    return urllib.request.urlopen(req)


def _file_links(token=None):
    """{filename: (download url, size)} of the record: published record first, else the draft (needs a token)."""
    try:
        with _get(f"{API}/records/{RECORD_ID}/files") as r:
            entries = json.load(r)["entries"]
        return {e["key"]: (e["links"]["content"], e["size"]) for e in entries}, None
    except urllib.error.HTTPError as err:
        if err.code not in (403, 404, 410):
            raise
    if token is None:
        raise RuntimeError(f"Zenodo record {RECORD_ID} is not public yet; set ZENODO_TOKEN (or ~/.zenodo_token) "
                           "to download from the draft")
    with _get(f"{API}/deposit/depositions/{RECORD_ID}/files", token) as r:
        files = json.load(r)
    return {f["filename"]: (f["links"]["download"], f["filesize"]) for f in files}, token


def _download(url, dest, size, token=None, chunk=1 << 22):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _get(url, token) as r, open(tmp, "wb") as f:
        done = 0
        while block := r.read(chunk):
            f.write(block)
            done += len(block)
            print(f"\r  {dest.name}: {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(systems=None, dest=None, keep_archives=False, force=False):
    """Download and unpack the archives of ``systems`` (default: all) into ``dest`` (default: data_root())."""
    systems = list(SYSTEMS if systems is None else [systems] if isinstance(systems, str) else systems)
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        raise ValueError(f"unknown system(s) {sorted(unknown)}; available: {SYSTEMS} "
                         "(adenylate_kinase: clone PSAnalysisTutorial, see data/README.md)")
    dest = Path(dest or data_root())
    dest.mkdir(parents=True, exist_ok=True)
    links, token = _file_links(_token())
    with _get(links["SHA256SUMS"][0], token) as r:
        sums = dict(line.split()[::-1] for line in r.read().decode().splitlines() if line.strip())
    for name in systems:
        if (dest / name).is_dir() and not force:
            print(f"{name}: already in {dest}")
            continue
        tar = dest / f"{name}.tar"
        url, size = links[tar.name]
        if not (tar.exists() and tar.stat().st_size == size):
            _download(url, tar, size, token)
        if _sha256(tar) != sums[tar.name]:
            raise RuntimeError(f"{tar}: checksum does not match SHA256SUMS; delete it and retry")
        if force and (dest / name).is_dir():
            shutil.rmtree(dest / name)
        with tarfile.open(tar) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:                       # Python without extraction filters
                t.extractall(dest)
        if not keep_archives:
            tar.unlink()
        print(f"{name}: unpacked into {dest / name}")
    return dest


def require(system_dir, download=True):
    """Make sure the data folder of one system exists (download it from Zenodo if not)."""
    system_dir = Path(system_dir)
    if system_dir.is_dir():
        return system_dir
    if not download or system_dir.name not in SYSTEMS:
        raise FileNotFoundError(f"{system_dir} not found; run `python scripts/download_data.py {system_dir.name}` "
                                "or see data/README.md")
    print(f"{system_dir.name}: not found in {system_dir.parent}, downloading from Zenodo ({DOI})")
    fetch([system_dir.name], dest=system_dir.parent)
    return system_dir
