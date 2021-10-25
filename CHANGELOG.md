# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.1.1] - working version
## Added
 - drivers: Sketch-and-Precondition for Saddle point systems (SPS2)
## Changed
 - drivers: class names for least squares
    * "SPO": Sketch-and-Precondition for Overdetermined least squares
    * SAS1 --> SPO3 (name used in the design doc)
    * SAS2 --> SPO1 (name used in the design doc)

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