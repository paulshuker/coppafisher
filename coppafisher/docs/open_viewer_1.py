from coppafisher import Notebook, Viewer

nb = Notebook("/path/to/notebook")
Viewer(nb, gene_marker_filepath="/path/to/custom/gene_marker_file.csv")
