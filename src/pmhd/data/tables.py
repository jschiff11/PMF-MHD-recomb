"""
Location of the Lyman-alpha moment tables: {a,b,c}barinterpmaster.pkl and f2bars_dict/.

Every consumer (FSR_TLA_Tfs*, cross_corr_and_source_f*, xe2*, single_mode_source, tests) reads the
tables through ``tables_dir()``, so an alternative table set (e.g. a rebuild with different physics)
can be used by exporting ``PMHD_TABLES_DIR`` without touching the canonical copy.  Other pre-stored
inputs (HyRec output, z-crossing caches, tau0arr) are NOT redirected.

Since 2026-09 the canonical ``pre_stored_data`` tables are the rebuilt ones (non-reflecting closure at
ell_max = 14 in all three towers, first-order Doppler l=1 source, Doppler/aberration keys).  The regression
tests compare against the published pre-2026-09 outputs, so they read the previous tables through
``reference_tables_dir()`` (``PMHD_REF_TABLES_DIR``, default ``pre_stored_data_prev``, not tracked; see "Regression
reference" in paper_data_plots/doppler_nrbc/VERIFICATION.md for how to recreate it from git).
"""
import os
from pathlib import Path

DEFAULT_TABLES_DIR = Path(__file__).resolve().parent / "pre_stored_data"


_CHECKED = set()


def check_tables_provenance(d):
    """If the table directory records its provenance (TABLES_PROVENANCE.json), require the constants set and the
    background it was built on to be the ones in use (PREREG addendum 17, item 3)."""
    d = Path(d)
    if d in _CHECKED or not (d / "TABLES_PROVENANCE.json").exists():
        return
    import json, hashlib
    from pmhd import cons
    prov = json.loads((d / "TABLES_PROVENANCE.json").read_text())
    if prov.get("codata_set") != cons.CODATA_SET:
        raise RuntimeError(f"tables in {d} were built with {prov.get('codata_set')}, the package uses {cons.CODATA_SET}")
    if os.environ.get("PMHD_XE_BG_OLD", "0") == "0":
        bg = DEFAULT_TABLES_DIR / prov["background_cache"]
        if hashlib.sha256(bg.read_bytes()).hexdigest() != prov["background_sha256"]:
            raise RuntimeError(f"tables in {d} were built on a different background than {bg}")
    _CHECKED.add(d)


def tables_dir():
    d = Path(os.environ.get("PMHD_TABLES_DIR", str(DEFAULT_TABLES_DIR)))
    check_tables_provenance(d)
    return d


PREV_TABLES_DIR = Path(__file__).resolve().parent / "pre_stored_data_prev"


def reference_tables_dir():
    """The pre-2026-09 moment tables (m=0 NRBC at ell_max = 8, m=+-1 hard truncation at ell_max = 29, no Doppler
    source) that the published paper_data_plots/data outputs were made with; used by the regression tests."""
    return Path(os.environ.get("PMHD_REF_TABLES_DIR", str(PREV_TABLES_DIR)))
