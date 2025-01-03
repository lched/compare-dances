import argparse
import os
import sys

import bpy


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def rm_extension(fname):
    return os.path.splitext(fname)[0]


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument("input_file", type=str)
    parser.add_argument(
        "output_folder",
        type=str,
    )
    parser.add_argument(
        "--no-fbx", dest="export_fbx", default=True, action="store_false"
    )
    parser.add_argument(
        "--no-bvh", dest="export_bvh", default=True, action="store_false"
    )
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()

    def rm_extension(fname):
        return os.path.splitext(fname)[0]

    stem = rm_extension(os.path.basename(args.input_file))

    ### 1. Remove everything
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    ### 2. Open FBX (hard coded for now)
    bpy.ops.import_scene.fbx(
        filepath=args.input_file,
        use_custom_props=True,
        ignore_leaf_bones=True,
        use_custom_props_enum_as_string=True,
        automatic_bone_orientation=True,
        global_scale=1,
    )

    INDEX_OF_OBJECT = 1
    obj_name = bpy.data.objects[INDEX_OF_OBJECT].name

    # obj_name = "BadWeed"
    bpy.data.objects[INDEX_OF_OBJECT].name = obj_name
    bpy.data.armatures[0].name = "Armature"
    bpy.data.actions[0].name = "Dance"

    ### 3. Add a bone to the spine
    bpy.ops.object.editmode_toggle()
    bpy.ops.armature.select_all(action="DESELECT")
    bpy.ops.object.editmode_toggle()
    bone = bpy.data.objects[INDEX_OF_OBJECT].data.bones["Spine"]
    bone.select_head = True
    bpy.ops.object.editmode_toggle()

    bpy.ops.armature.extrude_move(
        ARMATURE_OT_extrude={"forked": False},
        TRANSFORM_OT_translate={
            "value": (0, 0, -0.05),
            "orient_type": "GLOBAL",
            "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "orient_matrix_type": "GLOBAL",
            "constraint_axis": (False, False, False),
            "mirror": False,
            "use_proportional_edit": False,
            "proportional_edit_falloff": "SMOOTH",
            "proportional_size": 1,
            "use_proportional_connected": False,
            "use_proportional_projected": False,
            "snap": False,
            "snap_elements": {"INCREMENT"},
            "use_snap_project": False,
            "snap_target": "CLOSEST",
            "use_snap_self": True,
            "use_snap_edit": True,
            "use_snap_nonedit": True,
            "use_snap_selectable": False,
            "snap_point": (0, 0, 0),
            "snap_align": False,
            "snap_normal": (0, 0, 0),
            "gpencil_strokes": False,
            "cursor_transform": False,
            "texture_space": False,
            "remove_on_cancel": False,
            "view2d_edge_pan": False,
            "release_confirm": False,
            "use_accurate": False,
            "use_automerge_and_split": False,
        },
    )

    bpy.ops.object.editmode_toggle()
    new_bone = bpy.data.objects[INDEX_OF_OBJECT].data.bones["Spine.001"]
    new_bone.name = "Hips"
    new_bone.select = True
    bpy.ops.object.editmode_toggle()
    bpy.ops.armature.switch_direction()
    bpy.context.active_bone.head = (0, 0, 0)
    bpy.context.active_bone.roll = 0

    ### 4. Then, set the Spine, the RightUpLeg and the LeftUpLeg bones as children of the newly created bone
    bpy.data.armatures[0].edit_bones["Spine"].parent = bpy.data.armatures[0].edit_bones[
        "Hips"
    ]
    bpy.data.armatures[0].edit_bones["LeftUpLeg"].parent = bpy.data.armatures[
        0
    ].edit_bones["Hips"]
    bpy.data.armatures[0].edit_bones["RightUpLeg"].parent = bpy.data.armatures[
        0
    ].edit_bones["Hips"]

    bpy.ops.object.editmode_toggle()
    # Add control Armature
    bpy.ops.object.armature_add(
        enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.data.objects["Armature"].name = "Control"

    bpy.ops.object.posemode_toggle()

    bpy.ops.pose.constraint_add(type="COPY_LOCATION")
    bpy.ops.pose.constraint_add(type="COPY_ROTATION")
    bpy.context.object.pose.bones["Bone"].constraints[
        "Copy Location"
    ].target = bpy.data.objects[obj_name]
    bpy.context.object.pose.bones["Bone"].constraints[
        "Copy Rotation"
    ].target = bpy.data.objects[obj_name]

    # Get length of original animation
    obj = bpy.data.objects[obj_name]
    fcurve = obj.animation_data.action.fcurves[0]
    animation_length = len(fcurve.keyframe_points)

    bpy.ops.nla.bake(
        frame_start=1,
        frame_end=animation_length,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        bake_types={"POSE"},
    )

    bpy.ops.object.posemode_toggle()
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[obj_name].select_set(True)
    bpy.ops.object.posemode_toggle()

    bone = bpy.data.objects[obj_name].pose.bones["Hips"]
    constraint = bone.constraints.new("CHILD_OF")
    constraint.target = bpy.data.objects["Control"]
    constraint.subtarget = "Bone"

    fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves
    for fc in fcurves:
        if "pose" not in fc.data_path:
            fcurves.remove(fc)

    bpy.ops.nla.bake(
        frame_start=1,
        frame_end=animation_length,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        bake_types={"POSE"},
    )

    # Remove control bone
    bpy.data.objects.remove(bpy.data.objects["Control"])
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[0].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[INDEX_OF_OBJECT]

    # Export BVH file
    bvh_path = os.path.join(args.output_folder, stem + "_fixed" + ".bvh")
    bpy.ops.export_anim.bvh(
        filepath=bvh_path,
        frame_start=1,
        frame_end=animation_length,
        root_transform_only=True,
    )

    fbx_path = os.path.join(args.output_folder, stem + "_fixed" + ".fbx")
    bpy.ops.export_scene.fbx(filepath=fbx_path, check_existing=True)
