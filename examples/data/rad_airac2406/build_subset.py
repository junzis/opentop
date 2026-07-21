"""Build the checked-in AIRAC 2406 VST example subset.

Run this script from the repository root after placing the source export under
``tmp/rad_data``.  The generated files preserve source records verbatim while
retaining only the union of five diverse EHAM-LIRF routes at odd flight levels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from opentop import rad

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tmp" / "rad_data"
OUTPUT = Path(__file__).resolve().parent

NNPT_SOURCE = SOURCE / "NavPoint_2406.nnpt"
ASE_SOURCE = SOURCE / "VST_2406.ase"
ARP_SOURCE = SOURCE / "VST_2406_Airports.arp"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    navpoints = rad.read_nnpt(NNPT_SOURCE).records
    segments = rad.read_ase(ASE_SOURCE, navpoints, layer="vst").records
    airports = rad.read_arp(ARP_SOURCE).records

    # The supplied management export describes raw field 2 == 1 segments as
    # odd-level orientations.  FL350 uses that orientation.  This extraction
    # deliberately does not infer semantics for any other raw code.
    odd_codes = {segment.raw_fields for segment in segments if segment.raw_field_2 == 1}
    schema = rad.AseCodeSchema(
        "airac-2406-vst-odd",
        "1",
        {
            code: rad.AseSemantics(
                rad.EdgeDirection.FORWARD,
                metadata={"flight_level_parity": "odd"},
            )
            for code in odd_codes
        },
    )
    graph = rad.graph_from_ase(
        segments,
        navpoints,
        decode=schema,
        strict=False,
    )
    airport_by_icao = rad.airport_index(airports)
    connected, source, target = rad.add_airport_connectors(
        graph,
        airport_by_icao["EHAM"],
        airport_by_icao["LIRF"],
        connector_count=12,
        maximum_distance_m=300_000.0,
    )
    routes = rad.RoutePlanner(connected).candidates(
        source,
        target,
        config=rad.RouteSelectionConfig(
            candidates=5,
            search_candidates=300,
            max_cost_ratio=1.5,
            max_distance_ratio=1.6,
            maximum_shared_edge_fraction=0.7,
            budget=rad.SearchBudget(max_expansions=1_000_000),
        ),
    )
    if len(routes) != 5:
        raise RuntimeError(f"expected five source routes, found {len(routes)}")

    selected_lines = {
        edge.provenance.line_number
        for route in routes
        for edge in route.edges
        if edge.provenance is not None
    }
    selected_segments = [
        segment
        for segment in segments
        if segment.provenance.line_number in selected_lines
    ]
    selected_point_ids = {
        point_id
        for segment in selected_segments
        for point_id in (segment.source_id, segment.target_id)
    }
    selected_points = [
        point for point in navpoints if point.point_id in selected_point_ids
    ]
    selected_airports = [
        airport for airport in airports if airport.icao in {"EHAM", "LIRF"}
    ]

    nnpt_lines = [str(len(selected_points))] + [
        point.provenance.raw_line for point in selected_points
    ]
    ase_lines = [segment.provenance.raw_line for segment in selected_segments]
    arp_lines = [airport.provenance.raw_line for airport in selected_airports]
    (OUTPUT / "airac2406_vst_subset.nnpt").write_text(
        "\n".join(nnpt_lines) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "airac2406_vst_subset.ase").write_text(
        "\n".join(ase_lines) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "airac2406_airports.arp").write_text(
        "\n".join(arp_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "airac_cycle": "2406",
        "case": "EHAM-LIRF at FL350",
        "source_files": {
            NNPT_SOURCE.name: sha256(NNPT_SOURCE),
            ASE_SOURCE.name: sha256(ASE_SOURCE),
            ARP_SOURCE.name: sha256(ARP_SOURCE),
        },
        "selection": {
            "orientation": "odd",
            "source_candidate_count": len(routes),
            "maximum_shared_edge_fraction": 0.7,
            "connector_count": 12,
            "maximum_connector_distance_m": 300_000.0,
        },
        "records": {
            "navpoints": len(selected_points),
            "segments": len(selected_segments),
            "airports": len(selected_airports),
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
