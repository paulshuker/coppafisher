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
    ("omp_intensity_histogram", 5, False),
    ("omp_min_intensity", 10, False),
    ("open_viewer_0", 7, False),
    ("open_viewer_1", 7, False),
    ("retrieve_notebook_config", 20, False),
    ("run_pipeline_0", 10, False),
    ("update_tile_dir", 7, False),
    ("zip_nb", 7, True),
    ("zip_nb_2", 7, True),
)
NB_REPLACE = '"/path/to/notebook"'
CONFIG_REPLACE = '"/path/to/config.ini"'
GENE_MARKER_REPLACE = '"/path/to/custom/gene_marker_file.csv"'
TEMP_DIRECTORY_REPLACE = '"/path/to/temporary/directory/"'
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
        ("view_intensity_histogram(nb)", "view_intensity_histogram(nb, show=False)"),
        ("Viewer(nb)", 'Viewer(nb, gene_marker_filepath="/path/to/custom/gene_marker_file.csv")'),
        ("Viewer(nb, ", "Viewer(nb, show=False, "),
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

    temp_directories = [tempfile.TemporaryDirectory("coppafisher")]

    for file_name, timeout, modifies_nb in TEST_FILE_NAMES:
        nb_path_copy = nb_path
        temp_dir = None
        if modifies_nb:
            # Make a copy of the notebook before trying our scripts on it, this way code that modifies the notebook
            # won't modify the original.
            temp_dir = tempfile.TemporaryDirectory("coppafisher_nb")
            temp_directories.append(temp_dir)
            nb_path_copy = os.path.join(temp_dir.name, "notebook")

        if not os.path.isdir(nb_path_copy):
            shutil.copytree(nb_path, nb_path_copy)

        script_file_path = os.path.join(docs_dir, file_name + ".py")
        if os.path.isdir(script_file_path):
            continue
        if not script_file_path.endswith(".py"):
            continue

        with open(script_file_path, "r") as file:
            code = "\n".join(file.readlines())

        # Ensure headless mode when testing.
        code = "import matplotlib\nmatplotlib.use('Agg')\n" + code

        for replace_content, replace_to in REPLACEMENTS.items():
            code = code.replace(replace_content, replace_to)
        # Replace any temp notebook paths with a real notebook path.
        if NB_REPLACE in code:
            code += "\ndel nb"
        code = code.replace(NB_REPLACE, f'r"{nb_path_copy}"')
        code = code.replace(CONFIG_REPLACE, f'r"{config_path}"')
        code = code.replace(GENE_MARKER_REPLACE, f'r"{gene_marker_path}"')
        code = code.replace(TEMP_DIRECTORY_REPLACE, f'r"{temp_directories[0].name}"')

        res = pool.apply_async(exec, [code])
        try:
            res.get(timeout=timeout)
        except (EOFError, multiprocessing.TimeoutError):
            # An EOFError occurs when there is the use of `input()` in the code being run.
            pass

    pool.terminate()
    pool.close()
    pool.join()
    [temp_dir.cleanup() for temp_dir in temp_directories]


@pytest.mark.integration
def test_doc_imports() -> None:
    """All documentation `import` code snippets are evaluated to check they work okay."""

    # Code snippets are in the form of
    # ```py (or python)
    # code here
    # ```
    #
    # or
    #
    # `#!python code here`

    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "docs")

    assert os.path.isdir(docs_dir)

    for entry in os.scandir(docs_dir):
        file_path = os.path.join(docs_dir, entry.name)
        if os.path.isdir(file_path):
            continue

        with open(file_path, "r") as file:
            docs_lines = file.readlines()
            within_code_snippet = False

        SINGLE_LINE_SNIPPET_START = "`#!python "
        MULTILINE_SNIPPET_START = "```py"

        for doc_line in docs_lines:
            if SINGLE_LINE_SNIPPET_START in doc_line:
                code_start_index = doc_line.index(SINGLE_LINE_SNIPPET_START) + len(SINGLE_LINE_SNIPPET_START)
                code_end_index = doc_line.index("`", code_start_index)
                if ";" in doc_line[code_start_index:]:
                    code_end_index = doc_line.index(";")

                code_str = doc_line[code_start_index:code_end_index]
                if " import " not in code_str:
                    continue

                exec(code_str)

        for doc_line in docs_lines:
            if not within_code_snippet and doc_line.strip().startswith(MULTILINE_SNIPPET_START):
                within_code_snippet = True
                continue

            if within_code_snippet and " import " in doc_line:
                if doc_line.split()[1].startswith("linestuffup"):
                    continue
                exec(doc_line.strip())

            if within_code_snippet and "```" in doc_line:
                within_code_snippet = False
                continue
