from coppafisher import Notebook
from coppafisher.results import export_to_pciseq

nb = Notebook("/path/to/notebook")
export_to_pciseq(nb, method, score_thresh, intensity_thresh)
