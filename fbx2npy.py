import glob
import os
import pickle
import sys
from pathlib import Path

import bpy
import numpy as np

TOL = 1e-6


def blender_del_everything():
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")

    # Delete all objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete all materials
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)

    # Delete all meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)


def rm_extension(fname):
    """Remove the file extension at the end of a string."""
    return os.path.splitext(fname)[0]


def get_const_rows(arr, tol=0):
    diffs = np.abs(arr - arr[:, 0][:, None])
    max_diffs = np.max(diffs, axis=1)
    return np.where(max_diffs <= tol)[0]


def fbx_to_numpy(fbx_file, out_folder):
    print(fbx_file)
    blender_del_everything()
    # Load the FBX file into Blender
    bpy.ops.import_scene.fbx(
        filepath=fbx_file,
        use_custom_props=True,
        ignore_leaf_bones=True,
        use_custom_props_enum_as_string=True,
        automatic_bone_orientation=True,
        global_scale=100,
    )

    # Get a reference to the object you want to access
    obj = bpy.data.objects[0]

    # Get a reference to the animation data for the object
    anim_data = obj.animation_data

    times = np.zeros(len(anim_data.action.fcurves[0].keyframe_points))
    data = np.zeros(
        (
            len(anim_data.action.fcurves),
            len(anim_data.action.fcurves[0].keyframe_points),
        )
    )
    # Access the keyframes of the animation data
    channels = []
    for idx, fcu in enumerate(anim_data.action.fcurves):
        channels.append((fcu.data_path, fcu.array_index))
        for j, keyframe in enumerate(fcu.keyframe_points):
            time = keyframe.co[0]
            value = keyframe.co[1]
            if idx == 0:
                times[j] = time
            data[idx, j] = value

    # Filter out useless channels
    const_rows_ind = get_const_rows(data, tol=TOL)

    filtered_data = np.zeros((data.shape[0] - len(const_rows_ind), data.shape[-1]))
    filtered_channels = []
    filtered_array_idx = 0
    for idx in range(len(channels)):
        if idx not in const_rows_ind:
            filtered_data[filtered_array_idx, :] = data[idx, :]
            filtered_channels.append(channels[idx])
            filtered_array_idx += 1

    # Save the results over 3 files
    stem = str(Path(fbx_file).stem)
    print(os.path.join(out_folder, stem + "_data.npy"))
    np.save(os.path.join(out_folder, stem + "_data.npy"), filtered_data)
    np.save(os.path.join(out_folder, stem + "_times.npy"), times)
    with open(os.path.join(out_folder, stem + "_channels.pkl"), "wb") as file:
        pickle.dump(filtered_channels, file)


if __name__ == "__main__":
    # TODO: possible improvement: generate only 1 channels file for each participant
    """Arg1 is input_folder, arg2 is output folder"""
    input_folder = sys.argv[-2]
    out_folder = sys.argv[-1]

    fbx_files = glob.glob(os.path.join(input_folder, "*.fbx"))
    for f in fbx_files:
        try:
            fbx_to_numpy(f, out_folder)
        except RuntimeError as e:
            print(e)
