import os
import tempfile

import numpy as np
import zarr

from coppafisher.utils import zarray


def test_convert_group_to_zip_store() -> None:
    rng = np.random.RandomState(0)

    data = rng.rand(6, 7).astype(np.float16)
    data_sub = rng.rand(3, 4, 5).astype(np.float32)
    temp_dir = tempfile.TemporaryDirectory("coppafisher")
    group_path = os.path.join(temp_dir.name, "group.zipgroup")
    group = zarr.open_group(group_path, "w-", zarr_version=2)
    group.zeros_like("data", data, dtype=data.dtype)
    group["data"][:] = data
    group.create_group("subgroup")
    group["subgroup"].zeros("data_sub", shape=data_sub.shape, dtype=data_sub.dtype)
    group["subgroup"]["data_sub"][:] = data_sub
    del group

    zarray.convert_group_to_zip_store(group_path, "")

    store = zarr.ZipStore(group_path, mode="r")
    group = zarr.open_group(store, zarr_version=2)

    assert isinstance(group.store, zarr.ZipStore)
    assert "data" in group
    assert np.allclose(group["data"][:], data)
    assert group["data"].dtype == data.dtype
    assert "subgroup" in group
    assert "data_sub" in group["subgroup"]
    assert np.allclose(group["subgroup"]["data_sub"][:], data_sub)
    assert group["subgroup"]["data_sub"].dtype == data_sub.dtype

    store.close()
    temp_dir.cleanup()

    # Test with a specified temporary directory.
    data = rng.rand(6, 7).astype(np.float16)
    data_sub = rng.rand(3, 4, 5).astype(np.float32)
    temp_dir = tempfile.TemporaryDirectory("coppafisher")
    temp_dir2 = tempfile.TemporaryDirectory("coppafisher")
    group_path = os.path.join(temp_dir.name, "group.zipgroup")
    group = zarr.open_group(group_path, "w-", zarr_version=2)
    group.zeros_like("data", data, dtype=data.dtype)
    group["data"][:] = data
    group.create_group("subgroup")
    group["subgroup"].zeros("data_sub", shape=data_sub.shape, dtype=data_sub.dtype)
    group["subgroup"]["data_sub"][:] = data_sub
    del group

    zarray.convert_group_to_zip_store(group_path, temp_dir2.name)

    store = zarr.ZipStore(group_path, mode="r")
    group = zarr.open_group(store, zarr_version=2)

    assert isinstance(group.store, zarr.ZipStore)
    assert "data" in group
    assert np.allclose(group["data"][:], data)
    assert group["data"].dtype == data.dtype
    assert "subgroup" in group
    assert "data_sub" in group["subgroup"]
    assert np.allclose(group["subgroup"]["data_sub"][:], data_sub)
    assert group["subgroup"]["data_sub"].dtype == data_sub.dtype

    store.close()
    temp_dir.cleanup()
    temp_dir2.cleanup()


def test_convert_array_to_zip_store() -> None:
    rng = np.random.RandomState(0)

    data = rng.rand(6, 7, 2).astype(np.int32)
    temp_dir = tempfile.TemporaryDirectory("coppafisher")
    array_path = os.path.join(temp_dir.name, "array.zarray")
    array = zarr.open_array(array_path, "w-", shape=data.shape, dtype=data.dtype)
    assert not isinstance(array.store, zarr.ZipStore)
    array[:] = data
    del array

    zarray.convert_array_to_zip_store(array_path, "")
    store = zarr.ZipStore(array_path, mode="r")
    array = zarr.open_array(store)
    assert (array[:] == data).all()
    assert array.dtype == data.dtype

    store.close()
    temp_dir.cleanup()

    # Test with a specified temporary directory.
    data = rng.rand(6, 7, 2).astype(np.int32)
    temp_dir = tempfile.TemporaryDirectory("coppafisher")
    temp_dir2 = tempfile.TemporaryDirectory("coppafisher")
    array_path = os.path.join(temp_dir.name, "array.zarray")
    array = zarr.open_array(array_path, "w-", shape=data.shape, dtype=data.dtype)
    assert not isinstance(array.store, zarr.ZipStore)
    array[:] = data
    del array

    zarray.convert_array_to_zip_store(array_path, temp_dir2.name)
    store = zarr.ZipStore(array_path, mode="r")
    array = zarr.open_array(store)
    assert (array[:] == data).all()
    assert array.dtype == data.dtype

    store.close()
    temp_dir.cleanup()
    temp_dir2.cleanup()
