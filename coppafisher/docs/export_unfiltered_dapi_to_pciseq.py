from coppafisher import Notebook
from coppafisher.results import export_pciseq_unfiltered_dapi_image

nb = Notebook("/path/to/notebook")
config_path = "/path/to/config.ini"
export_pciseq_unfiltered_dapi_image(nb, config_path, radius_norm_file=None)
