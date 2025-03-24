import matplotlib

from coppafisher.plot.results_viewer import exporter


def test_ExportTool2D() -> None:
    matplotlib.use("Agg")

    tool = exporter.ExportTool2D(show=False)

    global click_count
    click_count = 0

    def assert_click(tool_click: exporter.ExportTool2D) -> None:
        assert type(tool_click) is exporter.ExportTool2D
        assert tool == tool_click

        global click_count
        click_count += 1

    tool.on_click = assert_click
    tool.on_export_button_clicked()
    assert click_count == 1
    tool.close()
