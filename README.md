# Compare Dances


## Using FBX files
FBX files exported from the motion capture plugin of Unreal Engine have a hierarchy that not compatible with BVH: they don't have a root node, but three. The Blender script [blender_fix_hierarchy](./blender_fix_hierarchy.py) allows to fix that and should be called with:
    blender -b -P .\blender_fix_hierarchy.py -- ./path/to/fbx/file out_folder/

This fixes the hierarchy to have a single root bone, without changing anything else in the architecture or the motion.


## Ideas for improvements:
 [] Rotate the incoming skeleton so that it's always facing the "right" direction, even if spectator are not perfectly in front of the camrea.