# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.1.4] - working version
## Added
 - drivers: SPO combines the functionality from SPO1 and SPO3
## Removed
 - drivers: SPO1 and SPO3
 
# [0.1.3] - superseded on Dec 10, 2021
## Added
 - Procedural interfaces to drivers (mostly undocumented).
 - Tons of documentation.
## Changed
 - How tolerances are handled in EVD1.
## Removed
 - CUR2.

# [0.1.2] - superseded on Nov 7, 2021
## Added
 - drivers: CUR in the style of VM2015 (CUR1)
## Changed
 - Package name changes from rlapy to parla.
 - drivers: CURD1 is now CUR2.
 - drivers: SPO defaults to obtaining the preconditioner by QR, but has the option of using Cholesky.

# [0.1.1] - superseded on Nov 3, 2021
## Added
 - drivers: Sketch-and-Precondition for Saddle point systems (SPS1, SPS2, name used in design doc)
 - drivers: QB-backed eigenvalue decomposition (EVD1)
 - drivers: Nystrom-backed eigenvalue decomposition (EVD2)
 - drivers: one-sided ID by QRCP of a sketch (OSID1, OSID2)
 - drivers: two-sided ID in the style of VM2015 (TSID1)
 - drivers: CUR in the style of VM2015 (CURD1)
 - comps: sketchers/oblivious.py (object-oriented wrappers)
 - comps: subset selection based on QRCP of a sketch (ROCS1)
## Changed
 - drivers: class names for least squares
    * "SPO": Sketch-and-Precondition for Overdetermined least squares
    * SAS1 --> SPO (name used in the design doc)
    * SAS2 --> SPO1 (name used in the design doc)
 - drivers: least squares solvers return two values: a vector (for x) and a dict (for a log).
 - comps: renamed QBFactorizer to QBDecomposer (similarly for TestQBFactorizer)
 - comps: moved and renamed comps/sketchers.py to comps/sketchers/aware.py.

# [0.1.0] - superseded on Oct 25, 2021
## Added
 - drivers: sketch-and-solve for overdetermined least squares (SAS1)
 - drivers: sketch-and-precondition for overdetermined least squares (SAP1, SAP2)
 - drivers: QB-backed SVD (SVD1)
 - comps: subspace power method for sketching (RS1)
 - comps: rangefinder backed by power method (RF1)
 - comps: QB computational routines (QB1, QB2, QB3)
 - comps: slightly modified version of SciPy's LSQR (rlapy.comps.lsqr).
 - comps: LSQR-backed deterministic saddle point solver (PcSS2).
 - utils: linalg_wrappers.py, sketching.py, and stats.py
