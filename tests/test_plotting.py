import matplotlib.pyplot as plt
import pytest

from opentop import plotting


@pytest.fixture(autouse=True)
def isolated_style():
    with plt.rc_context():
        yield


def test_publication_style_uses_colorblind_palette():
    plotting.apply_publication_style()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    assert tuple(colors) == plotting.OKABE_ITO


def test_axes_and_panel_helpers_apply_consistent_style():
    fig, axes = plt.subplots(1, 2)
    plotting.style_axes(axes[0])
    plotting.add_panel_labels(axes)

    assert not axes[0].spines["top"].get_visible()
    assert [text.get_text() for axis in axes for text in axis.texts] == ["A", "B"]
    plt.close(fig)


def test_wind_key_has_semitransparent_background():
    fig, axis = plt.subplots()
    vectors = axis.quiver([0.0], [0.0], [20.0], [0.0])

    plotting.add_wind_vector_key(axis, vectors)

    alpha = axis.patches[-1].get_alpha()
    assert alpha is not None
    assert 0 < alpha < 1
    plt.close(fig)
