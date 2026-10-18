# AIRAC 2406 VST example subset

This directory contains a compact source-derived subset of the supplied
EUROCONTROL/NM AIRAC 2406 SAAM/Gasel export for OpenTOP development. The
network case is EHAM to LIRF at FL350. The altitude examples also use EBBR to
EDDK and EHAM to LFLC. The subset preserves the original NNPT, ASE, and ARP
records used by five diverse graph routes; no coordinates or waypoint
identifiers are invented.

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

`airac2406_city_pair_caps.flc2` preserves three source rules and their original
comments: EBBR to EDDK is capped at FL175 (`03_EB4012`); EBAW to EHRD is capped
at FL55 (`03_EB4014`); EHAM to LFLC is capped at FL315 (`03_LF4137`). The
airport file also includes the original EBBR, EDDK, and LFLC reference records.
The notebook compares an A320 cruise from EBBR to EDDK with and without the
FL175 ceiling, using the same cached ERA5 wind. A full-flight section uses the
longer EHAM-LFLC pair (about 735 km) for OpenAP, BADA 3, and BADA 4, with and
without FL315. It uses matched initial mass, wind, and solver settings within
each model. BADA runs require licensed local data; configure
`OPENTOP_BADA3_PATH` and `OPENTOP_BADA4_PATH` as needed. These comparisons
isolate the vertical limits on each airport pair. They do not provide RAD
horizontal networks for those pairs.

The FLC2 file is a small source excerpt, not the complete restriction set. No
cap applies to EHAM-LIRF within this excerpt. This does not prove that the
complete RAD export has no restriction for that flight. Airport-group and
regional definitions, calendar conditions, and signed point/profile rules
remain outside the example. `require_resolved()` stops a solve when an unknown
city-pair rule could impose a tighter ceiling.
