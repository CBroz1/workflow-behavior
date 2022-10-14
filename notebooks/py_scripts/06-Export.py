# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.9.12 ('nwb')
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # DataJoint U24 - Workflow DeepLabCut

# %% [markdown]
# Change into the parent directory to find the `dj_local_conf.json` file. 

# %% tags=[]
import os
# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd())=='notebooks': os.chdir('..')

# %% [markdown]
# ## Write NWB file

# %%
from workflow_deeplabcut.pipeline import model, lab, session
from workflow_deeplabcut.export import dlc_session_to_nwb

session_key = (session.Session).fetch(limit=1)[0]
pose_key = (model.PoseEstimation).fetch("KEY")[0]
session_kwargs = dict(
    lab_key=(lab.Lab & "lab='LabA'").fetch1(),
    project_key=(lab.Protocol & "protocol='ProtA'").fetch1(),
    protocol_key=(lab.Project & "project='ProjA'").fetch1(),
)

# %%
session_kwargs

# %%
dlc_session_to_nwb(session_key, **session_kwargs)

# %%
fpath = dlc_session_to_nwb(pose_key,
                           use_element_session=True, 
                           session_kwargs=session_kwargs)

# %% [markdown]
# ## Bugfix work

# %%
from dlc2nwb.utils import get_movie_timestamps 
from element_interface.utils import find_full_path
from workflow_deeplabcut.paths import get_dlc_root_data_dir
vid_fp = str(find_full_path(get_dlc_root_data_dir(), 'from_top_tracking/videos/test.mp4'))
timestamps = get_movie_timestamps(str(vid_fp))

# %%
import numpy as np
if all(timestamps[-3:] == 0):
    avg_frame_diff = np.mean(np.diff(timestamps[:-3]))
    inferred_times = range(1,4) * avg_frame_diff + timestamps[-4]
    timestamps = np.concatenate((timestamps[:-3],inferred_times),axis=0)
print(timestamps[-10:])

# %%
import cv2
from deeplabcut.utils.auxfun_videos import VideoReader
reader = VideoReader(vid_fp)
timestamps = []
for _ in range(122):
    _ = reader.read_frame()
    timestamps.append(reader.video.get(cv2.CAP_PROP_POS_MSEC))

# %%
reader.video.get(cv2.CAP_PROP_POS_MSEC)

# %%
from dlc2nwb import utils
from element_interface.utils import find_full_path
from workflow_deeplabcut.paths import get_dlc_root_data_dir
from pathlib import Path
# import os; os.chdir('..')
from workflow_deeplabcut.pipeline import model, lab, session
from workflow_deeplabcut.export import dlc_session_to_nwb

session_key = (session.Session).fetch(limit=1)[0]
pose_key = (model.PoseEstimation).fetch("KEY")[0]
key = pose_key
write_file = True
subject_id = key["subject"]
output_dir = model.PoseEstimationTask.infer_output_dir(key)
CONFIGPATH = str(output_dir / "dj_dlc_config.yaml")
video_name = Path((model.VideoRecording.File & key).fetch1("file_path")).stem
FILEPATH = str(list(output_dir.glob(f"{video_name}*h5"))[0])

# %%
import os
import pandas as pd
from dlc2nwb import utils




def test_round_trip_conversion():
    df_ref = pd.read_hdf(FILEPATH)
    nwbfile = utils.convert_h5_to_nwb(
        CONFIGPATH,
        FILEPATH,
    )[0]
    df = utils.convert_nwb_to_h5(nwbfile).droplevel("individuals", axis=1)
    
    pd.testing.assert_frame_equal(df[:-3], df_ref[:-3])


def test_multi_animal_round_trip_conversion(tmp_path):
    dfs = []
    n_animals = 3
    for i in range(1, n_animals + 1):
        temp = utils._ensure_individuals_in_header(
            pd.read_hdf(FILEPATH),
            f"animal_{i}",
        )
        dfs.append(temp)
    df = pd.concat(dfs, axis=1)
    path_fake_df = str(tmp_path / os.path.split(FILEPATH)[1])
    df.to_hdf(path_fake_df, key="fake")
    nwbfiles = utils.convert_h5_to_nwb(CONFIGPATH, path_fake_df)
    assert len(nwbfiles) == n_animals


# %%
test_round_trip_conversion()

# %%
df[:-3]

# %%
df_ref[:-3]

# %% [markdown]
#
