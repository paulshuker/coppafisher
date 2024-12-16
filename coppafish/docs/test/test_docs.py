import subprocess
import tempfile
import time
from os import path

import pytest

TEST_FILE_NAMES = (
    "compatibility_tracker_stage_names",
    "compatibility_tracker_start_from",
    # "estimate_runtime",
    "export_to_pciseq_0",
    "export_to_pciseq_1",
    "generate_gene_codes",
    "retrieve_notebook_config",
)
NB_REPLACEMENT = '"/path/to/notebook"'
REPLACEMENTS = {
    "method": '"prob"',
    "n_rounds": "7",
    "n_channels": "9",
    "n_gene_codes": "50",
    "score_thresh": "0.05",
    "intensity_thresh": "0.05",
}


def run_python_script(script_path, timeout=10):
    # Start the process and run the Python script
    process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to complete, but give it a timeout (in seconds)
    start_time = time.time()
    while True:
        if process.poll() is not None:  # Check if process has completed
            break
        if time.time() - start_time > timeout:  # If the process is taking too long
            print("Timeout exceeded. Killing the process.")
            process.kill()  # Terminate the process
            return
        time.sleep(0.1)  # Sleep briefly to prevent busy-waiting

    # Get the output and errors from the process
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        assert False, f"Error occurred: {stderr.decode()}"
    else:
        print(f"Script output:\n{stdout.decode()}")


@pytest.mark.notebook
def test_all_docs() -> None:
    # Snippets of code used in the docs are integration tested to ensure that they do not crash when run. All code is
    # run with a timeout so code that hangs waiting for an input is not stuck forever.
    docs_dir = path.dirname(path.dirname(__file__))
    # TODO: Make a copy of the notebook before trying our scripts on it, this way code that modifies the notebook won't
    # modify the original.
    nb_path = path.join(
        path.dirname(docs_dir), "robominnie", "test", ".integration_dir", "output_coppafish", "notebook"
    )

    tmp_dir = tempfile.TemporaryDirectory(prefix="coppafish")

    for file_name in TEST_FILE_NAMES:
        full_path = path.join(docs_dir, file_name + ".py")
        if path.isdir(full_path):
            continue
        if not full_path.endswith(".py"):
            continue

        with open(full_path, "r") as file:
            code = "".join(file.readlines())

        # Replace any temp notebook paths with a real notebook path.
        code = code.replace(NB_REPLACEMENT, f'r"{nb_path}"')
        for replace_content, replace_to in REPLACEMENTS.items():
            code = code.replace(replace_content, replace_to)

        new_script_path = path.join(tmp_dir.name, "python_temp_run.py")
        with open(new_script_path, mode="w") as new_file:
            new_file.write(code)

        run_python_script(new_script_path)

    tmp_dir.cleanup()
