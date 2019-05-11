# -*- coding:utf-8 -*-
#
# test_trajectory_init.py

from mock import patch
import numpy as np
import pandas as pd
import pytest

from pbcluster import Trajectory

"""Tests for Trajectory initialization."""


@pytest.fixture()
def traj_mock_init():
    # See https://medium.com/
    #     @george.shuklin/mocking-complicated-init-in-python-6ef9850dd202
    with patch.object(Trajectory, "__init__", lambda x, y, z: None):
        return Trajectory(None, None)


def test__convert_ndarray_to_df_rejects_non_ndarrays(traj_mock_init):
    with pytest.raises(ValueError, match="must be a numpy ndarray"):
        non_ndarray = [[0, 1, 2], [3, 4, 5]]
        df = traj_mock_init._convert_ndarray_to_df(non_ndarray)


@pytest.mark.parametrize(
    "ndarray", [np.arange(3), np.arange(16).reshape((2, 2, 2, 2))]
)
def test__convert_ndarray_to_df_rejects_wrong_dimensions(
    ndarray, traj_mock_init
):
    with pytest.raises(ValueError, match="must have 2 or 3"):
        df = traj_mock_init._convert_ndarray_to_df(ndarray)


def test__convert_ndarray_to_df_converts_2d_array(traj_mock_init):
    array_2d = np.arange(6).reshape((2, 3))
    df = traj_mock_init._convert_ndarray_to_df(array_2d)
    assert len(df) == 2
    assert np.all(df["timestep"] == 0)
    assert np.allclose(array_2d, df[["x0", "x1", "x2"]].values)


def test__convert_ndarray_to_df_converts_3d_array(traj_mock_init):
    array_3d = np.arange(24).reshape((2, 4, 3))
    df = traj_mock_init._convert_ndarray_to_df(array_3d)
    print(df)
    print(df.loc[:4, "timestep"])
    assert len(df) == 8
    assert np.all(df.iloc[:4]["timestep"] == 0)
    assert np.all(df.iloc[4:]["timestep"] == 1)
    assert np.all(df.iloc[:4][["x0", "x1", "x2"]].values == array_3d[0, :, :])
    assert np.all(df.iloc[4:][["x0", "x1", "x2"]].values == array_3d[1, :, :])


@pytest.mark.parametrize(
    "input_df,error_text_match",
    [
        (
            pd.DataFrame(dict(x0=np.arange(3), x1=np.arange(3))),
            "particle_id column",
        ),
        (
            pd.DataFrame(dict(particle_id=np.arange(3), x1=np.arange(3))),
            "columns x0",
        ),
        (
            pd.DataFrame(dict(particle_id=np.ones(3), x0=np.arange(3))),
            "Duplicate particle_ids",
        ),
    ],
)
def test__verify_dataframe_raises_value_error(
    input_df, error_text_match, traj_mock_init
):
    with pytest.raises(ValueError, match=error_text_match):
        _ = traj_mock_init._verify_dataframe(input_df)


def test__verify_dataframe_fills_missing_timestep_and_particle_type_columns(
    traj_mock_init
):
    input_df = pd.DataFrame(dict(particle_id=np.arange(3), x0=np.arange(3)))
    output_df = traj_mock_init._verify_dataframe(input_df)
    assert "timestep" in output_df.columns
    assert "particle_type" in output_df.columns
    assert np.all(output_df["timestep"] == np.zeros(3))
    assert np.all(output_df["particle_type"] == np.zeros(3))


@pytest.mark.parametrize("x_names", [["x1"], ["x0", "x2"]])
def test__get_x_column_names_raises_error_on_bad_x_names(
    x_names, traj_mock_init
):
    with pytest.raises(ValueError, match="columns x0"):
        input_df = pd.DataFrame()
        for x_name in x_names:
            input_df[x_name] = np.arange(3)
        _ = traj_mock_init._get_x_column_names(input_df)


@pytest.mark.parametrize(
    "box_lengths,error,error_text_match",
    [
        ("foo", TypeError, "float or numpy array"),
        (np.array([1, 2]), ValueError, "must have length"),
        (-1, ValueError, "greater than 0"),
    ],
)
def test__verify_box_lengths_errors(
    box_lengths, error, error_text_match, traj_mock_init
):
    with pytest.raises(error, match=error_text_match):
        traj_mock_init.n_dimensions = 3
        _ = traj_mock_init._verify_box_lengths(box_lengths)
