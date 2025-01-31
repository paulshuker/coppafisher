import multiprocessing
import os
import shutil
import tempfile
from collections import OrderedDict

import pytest

# Each test file has its own timeout. This way the testing is not hanging around for tests that never finish without
# user input. The bool flag is set to true if the docs example modifies the notebook.
TEST_FILE_NAMES: tuple[tuple[str, int, bool]] = (
    ("compatibility_tracker_stage_names", 20, False),
    ("compatibility_tracker_start_from", 20, False),
    ("docstring_example", 5, False),
    ("estimate_runtime", 7, False),
    ("export_dapi_to_pciseq", 10, False),
    ("export_unfiltered_dapi_to_pciseq", 10, False),
    ("export_to_pciseq_0", 20, False),
    ("export_to_pciseq_1", 20, False),
    ("generate_gene_codes", 20, False),
    ("nb_delete_page_0", 7, True),
    ("nb_delete_page_1", 7, True),
    ("omp_min_intensity", 10, False),
    ("open_viewer_0", 7, False),
    ("open_viewer_1", 7, False),
    ("retrieve_notebook_config", 20, False),
    ("run_pipeline_0", 10, False),
)
NB_REPLACE = '"/path/to/notebook"'
CONFIG_REPLACE = '"/path/to/config.ini"'
GENE_MARKER_REPLACE = '"/path/to/custom/gene_marker_file.csv"'
REPLACEMENTS = OrderedDict(
    [
        ("method", '"prob"'),
        ("{tile}", "1"),
        ("n_rounds", "7"),
        ("n_channels", "9"),
        ("n_gene_codes", "11"),
        ("score_thresh", "0.05"),
        ("intensity_thresh", "0.05"),
        ("page_name", "omp"),
        ("Viewer(nb)", 'Viewer(nb, gene_marker_filepath="/path/to/custom/gene_marker_file.csv")'),
        ("Viewer(nb, ", "import matplotlib\nmatplotlib.use('Agg')\nViewer(nb, show=False, "),
    ]
)


@pytest.mark.notebook
def test_all_docs() -> None:
    # Snippets of code used in the docs are integration tested to ensure that they do not crash when run. All code is
    # run with a timeout so code that hangs waiting for an input (or takes a long compute time) is not stuck forever.
    docs_dir = os.path.dirname(os.path.dirname(__file__))

    config_path = os.path.join(os.path.dirname(docs_dir), "robominnie", "test", ".integration_dir", "robominnie.ini")
    gene_marker_path = os.path.join(
        os.path.dirname(docs_dir), "robominnie", "test", ".integration_dir", "gene_colours.csv"
    )
    nb_path = os.path.join(os.path.dirname(config_path), "output_coppafisher", "notebook")

    pool = multiprocessing.Pool(1)

    with tempfile.TemporaryDirectory(suffix="coppafisher_nb") as temp_dir:
        nb_path_copy = os.path.join(temp_dir, "notebook_copy")
        shutil.copytree(nb_path, nb_path_copy)

        last_test_modified_nb = False
        for file_name, timeout, modifies_nb in TEST_FILE_NAMES:
            if last_test_modified_nb:
                # Make a copy of the notebook before trying our scripts on it, this way code that modifies the notebook
                # won't modify the original.
                if os.path.isdir(nb_path_copy):
                    shutil.rmtree(nb_path_copy)
                shutil.copytree(nb_path, nb_path_copy)

            script_file_path = os.path.join(docs_dir, file_name + ".py")
            if os.path.isdir(script_file_path):
                continue
            if not script_file_path.endswith(".py"):
                continue

            with open(script_file_path, "r") as file:
                code = "\n".join(file.readlines())

            for replace_content, replace_to in REPLACEMENTS.items():
                code = code.replace(replace_content, replace_to)
            # Replace any temp notebook paths with a real notebook path.
            code = code.replace(NB_REPLACE, f'r"{nb_path_copy}"')
            code = code.replace(CONFIG_REPLACE, f'r"{config_path}"')
            code = code.replace(GENE_MARKER_REPLACE, f'r"{gene_marker_path}"')

            res = pool.apply_async(exec, [code])
            try:
                res.get(timeout=timeout)
            except (EOFError, multiprocessing.TimeoutError):
                # An EOFError occurs when there is the use of `input()` in the code being run.
                pass

            last_test_modified_nb = modifies_nb
