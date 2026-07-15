import warnings

from opentop._performance import build_performance_models


def test_openap_model_build_suppresses_wave_drag_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_performance_models(
            "A320",
            engine=None,
            use_synonym=False,
            performance_model="openap",
        )

    assert not any("Wave drag is experimental" in str(item.message) for item in caught)
