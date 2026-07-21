import matplotlib.pyplot as plt

from opentop import plotting


def test_publication_style_uses_colorblind_palette():
    plotting.apply_publication_style()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    assert tuple(colors) == plotting.OKABE_ITO
    assert plt.rcParams["savefig.dpi"] == 300


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

    assert axis.patches[-1].get_alpha() == 0.8
    plt.close(fig)
