"""Tests for normalized RAD network ingestion."""

from pathlib import Path

import pytest

from opentop import rad

FIXTURES = Path(__file__).parent / "fixtures" / "rad"


def test_nnpt_and_ase_resolve_underscored_endpoint_ids():
    navpoint_result = rad.read_nnpt(FIXTURES / "network.nnpt")
    ase_result = rad.read_ase(
        FIXTURES / "segments.ase",
        navpoint_result.records,
        layer="vst",
    )

    assert navpoint_result.status is rad.Conformance.VALID
    assert ase_result.status is rad.Conformance.VALID
    assert len(ase_result.records) == 3
    assert ase_result.records[0].source_id == "AAA"
    assert ase_result.records[0].target_id == "A_B"
    assert ase_result.records[1].source_id == "A_B"
    assert ase_result.records[1].target_id == "C"
    assert ase_result.records[0].source_latitude == pytest.approx(50.0)
    assert ase_result.records[0].raw_fields == (0, 1, 10)


def test_ase_coordinate_mismatch_is_indeterminate(tmp_path):
    navpoints = rad.read_nnpt(FIXTURES / "network.nnpt").records
    ase_path = tmp_path / "mismatch.ase"
    ase_path.write_text("0 1 10 0.0 120.0 3030.0 180.0 AAA_A_B\n", encoding="utf-8")

    result = rad.read_ase(ase_path, navpoints)

    assert result.status is rad.Conformance.INDETERMINATE
    assert not result.records
    assert result.diagnostics[0].code == "coordinate_mismatch"


def test_invalid_ase_record_fails_closed_in_strict_mode(tmp_path):
    navpoints = rad.read_nnpt(FIXTURES / "network.nnpt").records
    ase_path = tmp_path / "invalid.ase"
    ase_path.write_text("not an ASE record\n", encoding="utf-8")

    with pytest.raises(rad.RadParseError, match="eight fields"):
        rad.read_ase(ase_path, navpoints)

    result = rad.read_ase(ase_path, navpoints, strict=False)
    assert result.status is rad.Conformance.INVALID


def test_arp_coordinates_are_converted_from_minutes():
    result = rad.read_arp(FIXTURES / "airports.arp")

    assert result.records[0].icao == "AAAA"
    assert result.records[0].latitude == pytest.approx(50.0)
    assert result.records[0].longitude == pytest.approx(2.0)
    assert result.records[1].fir_id is None


def test_rts_records_are_sorted_and_build_known_route_edges():
    navpoints = rad.read_nnpt(FIXTURES / "network.nnpt").records
    routes = rad.read_routes(FIXTURES / "routes.rts").records

    graph = rad.graph_from_routes(routes, navpoints)

    assert [point.sequence for point in routes] == [1, 2, 3, 4]
    assert len(graph.edges) == 3
    assert [edge.target for edge in graph.outgoing("AAA")] == ["A_B"]


def test_ase_graph_requires_explicit_semantic_decoder():
    navpoints = rad.read_nnpt(FIXTURES / "network.nnpt").records
    segments = rad.read_ase(FIXTURES / "segments.ase", navpoints).records

    def decode(segment):
        if segment.raw_fields == (0, 1, 10):
            return rad.AseSemantics(rad.EdgeDirection.FORWARD)
        return None

    with pytest.raises(rad.UnsupportedAseCode, match="does not support"):
        rad.graph_from_ase(segments, navpoints, decode=decode)

    graph = rad.graph_from_ase(segments, navpoints, decode=decode, strict=False)

    assert len(graph.edges) == 2
    assert not list(graph.outgoing("C"))


def test_versioned_ase_code_schema_maps_only_verified_codes():
    navpoints = rad.read_nnpt(FIXTURES / "network.nnpt").records
    segments = rad.read_ase(FIXTURES / "segments.ase", navpoints).records
    schema = rad.AseCodeSchema(
        "fixture",
        "1",
        {(0, 1, 10): rad.AseSemantics(rad.EdgeDirection.FORWARD)},
    )

    assert schema.schema_id == "fixture:1"
    assert schema(segments[0]) is not None
    assert schema(segments[-1]) is None
