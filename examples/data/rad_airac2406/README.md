# AIRAC 2406 VST example subset

This directory contains a compact source-derived subset of the supplied
EUROCONTROL/NM AIRAC 2406 SAAM/Gasel export for OpenTOP development. The case is
EHAM to LIRF at FL350. It preserves the original NNPT, ASE, and ARP records used
by five diverse graph routes; no coordinates or waypoint identifiers are
invented.

The subset is intentionally a development example, not a current operational
flight-planning product. It covers the VST topology and the odd-flight-level
orientation used by the example. It does not claim complete evaluation of every
RAD management rule, time restriction, aerodrome procedure, or current AIRAC
change.

`build_subset.py` reproduces the checked-in files from source data placed under
`tmp/rad_data`. `manifest.json` records the source filenames, SHA-256 hashes,
selection settings, and output record counts.

Source: EUROCONTROL Network Manager AIRAC 2406 data, used with the repository
owner's permission for OpenTOP development.
