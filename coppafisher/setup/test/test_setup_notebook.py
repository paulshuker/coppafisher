import os
import shutil
import tempfile
from pathlib import PurePath

import numpy as np
import zarr

from coppafisher import utils
from coppafisher.setup.notebook import Notebook
from coppafisher.setup.notebook_page import NotebookPage, PageTypeError
from coppafisher.utils import system


def test_Notebook() -> None:
    rng = np.random.RandomState(0)

    nb_temp_dir = tempfile.TemporaryDirectory("coppafisher_nb")
    nb_path = os.path.join(nb_temp_dir.name, ".notebook_test")
    if os.path.isdir(nb_path):
        shutil.rmtree(nb_path)
    config_path = os.path.abspath("dslkhgdsjlgh")
    nb = Notebook(nb_path, config_path, must_exist=False)
    assert len(nb.get_all_versions()) == 0

    assert not nb.has_page("debug")
    assert nb.config_path == config_path

    nb_page: NotebookPage = NotebookPage("debug")

    a = 5
    b = 5.0
    c = True
    d = (4, 5, 6, 7)
    e = ((0.4, 5.0), (2.0, 1.0, 4.5), tuple())
    f = 3.0
    g = None
    h = 4.3
    i = "Hello, World"
    j = rng.rand(3, 10).astype(dtype=np.float16)
    k = rng.randint(2000, size=10, dtype=np.int64)
    l = rng.randint(2, size=(3, 4, 6), dtype=bool)
    m = np.zeros(3, dtype=str)
    m[0] = "blah"
    n = rng.randint(200, size=(7, 8), dtype=np.uint32)

    def _check_variables(nb: Notebook):
        assert nb.has_page("debug")
        assert np.allclose(nb.debug.a, a)
        assert np.allclose(nb.debug.b, b)
        assert np.allclose(nb.debug.c, c)
        assert np.allclose(nb.debug.d, d)
        assert type(nb.debug.d) is list
        assert nb.debug.e == utils.base.deep_convert(e, list)
        assert type(nb.debug.e) is list
        assert np.allclose(nb.debug.f, f)
        assert nb.debug.g is g
        assert np.allclose(nb.debug.h, h)
        assert nb.debug.i == i
        assert np.allclose(nb.debug.j, j)
        assert np.allclose(nb.debug.k, k)
        assert np.allclose(nb.debug.l, l)
        assert (nb.debug.m == m).all()
        assert np.allclose(nb.debug.n, n)
        zarray_path = os.path.abspath(nb.debug.o.store.path)
        assert os.path.isdir(zarray_path)
        assert len(os.listdir(zarray_path)) > 1
        assert PurePath(nb_path) in PurePath(zarray_path).parents
        zgroup_path = os.path.abspath(nb.debug.p.store.path)
        assert os.path.isdir(zgroup_path)
        assert len(os.listdir(zgroup_path)) > 0
        assert type(nb.debug.p["subgroup"]) is zarr.Group
        assert type(nb.debug.p["subarray.zarr"]) is zarr.Array
        assert nb.debug.p["subarray.zarr"].shape == (10, 5)

    nb_page.a = a
    try:
        nb_page.b = 5
        raise AssertionError("Should not be able to set a float type to an int")
    except PageTypeError:
        pass
    nb_page.b = b
    nb_page.c = c
    try:
        nb_page.d = (5, "4", True)
        nb_page.d = (5, "4")
        nb_page.d = (5, 0.5)
        raise AssertionError("Should not be able to set a tuple[int] type like this")
    except PageTypeError:
        pass
    nb_page.d = tuple()
    del nb_page.d
    nb_page.d = d
    nb_page.e = e
    nb_page.f = 3
    del nb_page.f
    nb_page.f = f
    nb_page.g = g
    nb_page.h = None
    del nb_page.h
    nb_page.h = h
    nb_page.i = i
    try:
        nb_page.j = np.zeros(10, dtype=int)
        raise AssertionError("Should not be able to set a ndarray[float] type like this")
    except PageTypeError:
        pass
    try:
        nb_page.j = np.zeros(10, dtype=bool)
        raise AssertionError("Should not be able to set a ndarray[float] type like this")
    except PageTypeError:
        pass
    nb_page.j = j
    try:
        nb_page.k = np.zeros(10, dtype=np.float32)
        raise AssertionError("Should not be able to set a ndarray[int] type like this")
    except PageTypeError:
        pass
    try:
        nb_page.k = zarr.array(np.ones(10))
        raise AssertionError("Should not be able to set a ndarray[int] type to zarray")
    except PageTypeError:
        pass
    try:
        nb_page.k = np.zeros(10, dtype=bool)
        raise AssertionError("Should not be able to set a ndarray[int] type like this")
    except PageTypeError:
        pass
    try:
        nb_page.k = False
        raise AssertionError("Should not be able to set a ndarray[int] type like this")
    except PageTypeError:
        pass
    nb_page.k = k
    nb_page.l = l

    nb_page.m = m
    nb_page.n = n

    try:
        nb += nb_page
        raise AssertionError("Should crash when adding an unfinished notebook page")
    except ValueError:
        pass

    temp_zarr = tempfile.TemporaryDirectory()
    array_saved = np.zeros((4, 8), dtype=np.float32)
    zarr_array_temp = zarr.open_array(
        store=temp_zarr.name, shape=array_saved.shape, dtype="|f4", zarr_version=2, chunks=(2, 4), mode="w"
    )
    zarr_array_temp[:] = array_saved.copy()

    assert nb_page.get_unset_variables() == ("o", "p")

    nb_page.o = zarr_array_temp
    del zarr_array_temp

    assert len(nb_page.get_unset_variables()) == 1
    assert nb_page.name == "debug"

    try:
        nb += nb_page
        raise AssertionError("Should not be able to add an unfinished page to the notebook")
    except ValueError:
        pass

    temp_zgroup = tempfile.TemporaryDirectory()
    group = zarr.group(store=temp_zgroup.name, zarr_version=2)
    group.create_dataset("subarray.zarr", shape=(10, 5), dtype=np.int16)
    group.create_group("subgroup")
    nb_page.p = group

    nb += nb_page

    try:
        nb.fake_variable = 4
        raise AssertionError("Should not be able to add integer variables to the notebook")
    except TypeError:
        pass
    try:
        nb += nb_page
        raise AssertionError("Should not be able to add the same page twice")
    except ValueError:
        pass

    _ = nb > "debug"
    _ = nb_page > "o"

    assert len(nb.get_all_versions()) == 1
    assert nb.get_all_versions()["debug"] == system.get_software_version()
    _check_variables(nb)
    del nb_page
    nb.resave()
    _check_variables(nb)
    del nb.debug.a
    a = 10
    nb.debug.a = a
    nb.resave()
    _check_variables(nb)

    del nb
    nb = Notebook(nb_path, must_exist=True)
    _check_variables(nb)

    nb._prune_locations = {"debug": ("o.zarray",)}
    _check_variables(nb)
    nb.prune(prompt=False)
    try:
        _check_variables(nb)
        AssertionError("Excepted prune to cause an error when checking variable o")
    except AssertionError:
        pass

    # Check that the resave function can safely remove pages.
    del nb.debug
    nb.resave()
    assert not nb.has_page("debug")
    assert not os.path.exists(os.path.join(nb_path, "debug"))

    nb = Notebook(nb_path)
    assert not nb.has_page("debug")
    assert not os.path.exists(os.path.join(nb_path, "debug"))

    # Clean any temporary files/directories.
    nb_temp_dir.cleanup()
    temp_zarr.cleanup()
    temp_zgroup.cleanup()
