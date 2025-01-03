from DisplayCommon import *
from fbx import *


def getAnimationData(pScene):
    sceneData = {}
    for i in range(
        pScene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))
    ):
        lAnimStack = pScene.GetSrcObject(
            FbxCriteria.ObjectType(FbxAnimStack.ClassId), i
        )

        lOutputString = "Animation Stack Name: "
        lOutputString += lAnimStack.GetName()
        lOutputString += "\n"
        print(lOutputString)

        getAnimationStackData(lAnimStack, pScene.GetRootNode(), True)
        getAnimationStackData(lAnimStack, pScene.GetRootNode(), False)


def getAnimationStackData(pAnimStack, pNode, isSwitcher):
    nbAnimLayers = pAnimStack.GetSrcObjectCount(
        FbxCriteria.ObjectType(FbxAnimLayer.ClassId)
    )

    lOutputString = "Animation stack contains "
    lOutputString += str(nbAnimLayers)
    lOutputString += " Animation Layer(s)"
    print(lOutputString)
    animationStackData = []

    for l in range(nbAnimLayers):
        lAnimLayer = pAnimStack.GetSrcObject(
            FbxCriteria.ObjectType(FbxAnimLayer.ClassId), l
        )

        # lOutputString = "AnimLayer "
        # lOutputString += str(l)
        # print(lOutputString)

        animationStackData.append(GetAnimationLayerData(lAnimLayer, pNode, isSwitcher))
    return animationStackData


def GetAnimationLayerData(pAnimLayer, pNode, isSwitcher=False):
    animationLayerData = []
    # lOutputString = "     Node Name: "
    # lOutputString += pNode.GetName()
    # lOutputString += "\n"
    # print(lOutputString)

    result = GetChannelsData(
        pNode, pAnimLayer, DisplayCurveKeys, GetListCurveKeys, isSwitcher
    )
    print

    for lModelCount in range(pNode.GetChildCount()):
        GetAnimationLayerData(pAnimLayer, pNode.GetChild(lModelCount), isSwitcher)


def GetChannelsData(pNode, pAnimLayer, DisplayCurve, DisplayListCurve, isSwitcher):
    result = {}

    def add_to_result(category, key, curve):
        if category not in result:
            result[category] = {}
        result[category][key] = DisplayCurve(curve) if curve else None

    KFCURVENODE_T_X = "X"
    KFCURVENODE_T_Y = "Y"
    KFCURVENODE_T_Z = "Z"

    KFCURVENODE_R_X = "X"
    KFCURVENODE_R_Y = "Y"
    KFCURVENODE_R_Z = "Z"

    KFCURVENODE_S_X = "X"
    KFCURVENODE_S_Y = "Y"
    KFCURVENODE_S_Z = "Z"

    if not isSwitcher:
        for axis, name in [
            (KFCURVENODE_T_X, "TX"),
            (KFCURVENODE_T_Y, "TY"),
            (KFCURVENODE_T_Z, "TZ"),
        ]:
            lAnimCurve = pNode.LclTranslation.GetCurve(pAnimLayer, axis)
            add_to_result("Translation", name, lAnimCurve)

        for axis, name in [
            (KFCURVENODE_R_X, "RX"),
            (KFCURVENODE_R_Y, "RY"),
            (KFCURVENODE_R_Z, "RZ"),
        ]:
            lAnimCurve = pNode.LclRotation.GetCurve(pAnimLayer, axis)
            add_to_result("Rotation", name, lAnimCurve)

        for axis, name in [
            (KFCURVENODE_S_X, "SX"),
            (KFCURVENODE_S_Y, "SY"),
            (KFCURVENODE_S_Z, "SZ"),
        ]:
            lAnimCurve = pNode.LclScaling.GetCurve(pAnimLayer, axis)
            add_to_result("Scaling", name, lAnimCurve)

    lNodeAttribute = pNode.GetNodeAttribute()
    KFCURVENODE_COLOR_RED = "X"
    KFCURVENODE_COLOR_GREEN = "Y"
    KFCURVENODE_COLOR_BLUE = "Z"

    if lNodeAttribute:
        for color, name in [
            (KFCURVENODE_COLOR_RED, "Red"),
            (KFCURVENODE_COLOR_GREEN, "Green"),
            (KFCURVENODE_COLOR_BLUE, "Blue"),
        ]:
            lAnimCurve = lNodeAttribute.Color.GetCurve(pAnimLayer, color)
            add_to_result("Color", name, lAnimCurve)

        light = pNode.GetLight()
        if light:
            for attr, name in [
                (light.Intensity, "Intensity"),
                (light.OuterAngle, "Cone Angle"),
                (light.Fog, "Fog"),
            ]:
                lAnimCurve = attr.GetCurve(pAnimLayer)
                add_to_result("Light", name, lAnimCurve)

        camera = pNode.GetCamera()
        if camera:
            for attr, name in [
                (camera.FieldOfView, "Field of View"),
                (camera.FieldOfViewX, "Field of View X"),
                (camera.FieldOfViewY, "Field of View Y"),
                (camera.OpticalCenterX, "Optical Center X"),
                (camera.OpticalCenterY, "Optical Center Y"),
                (camera.Roll, "Roll"),
            ]:
                lAnimCurve = attr.GetCurve(pAnimLayer)
                add_to_result("Camera", name, lAnimCurve)

        if lNodeAttribute.GetAttributeType() in (
            FbxNodeAttribute.eMesh,
            FbxNodeAttribute.eNurbs,
            FbxNodeAttribute.ePatch,
        ):
            lGeometry = lNodeAttribute
            lBlendShapeDeformerCount = lGeometry.GetDeformerCount(
                FbxDeformer.eBlendShape
            )
            for lBlendShapeIndex in range(lBlendShapeDeformerCount):
                lBlendShape = lGeometry.GetDeformer(
                    lBlendShapeIndex, FbxDeformer.eBlendShape
                )
                lBlendShapeChannelCount = lBlendShape.GetBlendShapeChannelCount()
                for lChannelIndex in range(lBlendShapeChannelCount):
                    lChannel = lBlendShape.GetBlendShapeChannel(lChannelIndex)
                    lChannelName = lChannel.GetName()
                    lAnimCurve = lGeometry.GetShapeChannel(
                        lBlendShapeIndex, lChannelIndex, pAnimLayer, True
                    )
                    add_to_result("Geometry", f"Shape {lChannelName}", lAnimCurve)

    lProperty = pNode.GetFirstProperty()
    while lProperty.IsValid():
        if lProperty.GetFlag(FbxPropertyFlags.eUserDefined):
            lFbxFCurveNodeName = lProperty.GetName()
            lCurveNode = lProperty.GetCurveNode(pAnimLayer)

            if not lCurveNode:
                lProperty = pNode.GetNextProperty(lProperty)
                continue

            lDataType = lProperty.GetPropertyDataType()
            category = "Properties"

            if lDataType.GetType() in {eFbxBool, eFbxDouble, eFbxFloat, eFbxInt}:
                for c in range(lCurveNode.GetCurveCount(0)):
                    lAnimCurve = lCurveNode.GetCurve(0, c)
                    add_to_result(category, lFbxFCurveNodeName, lAnimCurve)

            elif (
                lDataType.GetType() in {eFbxDouble3, eFbxDouble4}
                or lDataType.Is(FbxColor3DT)
                or lDataType.Is(FbxColor4DT)
            ):
                components = (
                    ["X", "Y", "Z"]
                    if not lDataType.Is(FbxColor3DT)
                    else ["Red", "Green", "Blue"]
                )
                for i, component in enumerate(components):
                    for c in range(lCurveNode.GetCurveCount(i)):
                        lAnimCurve = lCurveNode.GetCurve(i, c)
                        add_to_result(
                            category, f"{lFbxFCurveNodeName} {component}", lAnimCurve
                        )

            elif lDataType.GetType() == eFbxEnum:
                for c in range(lCurveNode.GetCurveCount(0)):
                    lAnimCurve = lCurveNode.GetCurve(0, c)
                    add_to_result(category, lFbxFCurveNodeName, lAnimCurve)

        lProperty = pNode.GetNextProperty(lProperty)

    return result


def InterpolationFlagToIndex(flags):
    # if (flags&KFCURVE_INTERPOLATION_CONSTANT)==KFCURVE_INTERPOLATION_CONSTANT:
    #    return 1
    # if (flags&KFCURVE_INTERPOLATION_LINEAR)==KFCURVE_INTERPOLATION_LINEAR:
    #    return 2
    # if (flags&KFCURVE_INTERPOLATION_CUBIC)==KFCURVE_INTERPOLATION_CUBIC:
    #    return 3
    return 0


def ConstantmodeFlagToIndex(flags):
    # if (flags&KFCURVE_CONSTANT_STANDARD)==KFCURVE_CONSTANT_STANDARD:
    #    return 1
    # if (flags&KFCURVE_CONSTANT_NEXT)==KFCURVE_CONSTANT_NEXT:
    #    return 2
    return 0


def TangeantmodeFlagToIndex(flags):
    # if (flags&KFCURVE_TANGEANT_AUTO) == KFCURVE_TANGEANT_AUTO:
    #    return 1
    # if (flags&KFCURVE_TANGEANT_AUTO_BREAK)==KFCURVE_TANGEANT_AUTO_BREAK:
    #    return 2
    # if (flags&KFCURVE_TANGEANT_TCB) == KFCURVE_TANGEANT_TCB:
    #    return 3
    # if (flags&KFCURVE_TANGEANT_USER) == KFCURVE_TANGEANT_USER:
    #    return 4
    # if (flags&KFCURVE_GENERIC_BREAK) == KFCURVE_GENERIC_BREAK:
    #    return 5
    # if (flags&KFCURVE_TANGEANT_BREAK) ==KFCURVE_TANGEANT_BREAK:
    #    return 6
    return 0


def TangeantweightFlagToIndex(flags):
    # if (flags&KFCURVE_WEIGHTED_NONE) == KFCURVE_WEIGHTED_NONE:
    #    return 1
    # if (flags&KFCURVE_WEIGHTED_RIGHT) == KFCURVE_WEIGHTED_RIGHT:
    #    return 2
    # if (flags&KFCURVE_WEIGHTED_NEXT_LEFT) == KFCURVE_WEIGHTED_NEXT_LEFT:
    #    return 3
    return 0


def TangeantVelocityFlagToIndex(flags):
    # if (flags&KFCURVE_VELOCITY_NONE) == KFCURVE_VELOCITY_NONE:
    #    return 1
    # if (flags&KFCURVE_VELOCITY_RIGHT) == KFCURVE_VELOCITY_RIGHT:
    #    return 2
    # if (flags&KFCURVE_VELOCITY_NEXT_LEFT) == KFCURVE_VELOCITY_NEXT_LEFT:
    #    return 3
    return 0


def DisplayCurveKeys(pCurve):
    # interpolation = ["?", "constant", "linear", "cubic"]
    lKeyCount = pCurve.KeyGetCount()
    CurveKeys = []

    for lCount in range(lKeyCount):
        lTimeString = ""
        lKeyValue = pCurve.KeyGetValue(lCount)
        lKeyTime = pCurve.KeyGetTime(lCount)

        data = {"Key Time": lKeyTime.GetTimeString(lTimeString), "Key Value": lKeyValue}
        # lOutputString = "            Key Time: "
        # lOutputString += lKeyTime.GetTimeString(lTimeString)
        # lOutputString += ".... Key Value: "
        # lOutputString += str(lKeyValue)
        # lOutputString += " [ "
        # lOutputString += interpolation[
        #     InterpolationFlagToIndex(pCurve.KeyGetInterpolation(lCount))
        # ]

        # lOutputString += " ]"
        # print(lOutputString)
        CurveKeys.append(data)
    return CurveKeys


def GetCurveDefault(pCurve):
    # lOutputString = "            Default Value: "
    # lOutputString += pCurve.GetValue()

    # print(lOutputString)
    return pCurve.GetValue()


def GetListCurveKeys(pCurve, pProperty):
    lKeyCount = pCurve.KeyGetCount()
    ListCurveKeys = []

    for lCount in range(lKeyCount):
        lKeyValue = static_cast < int > (pCurve.KeyGetValue(lCount))
        lKeyTime = pCurve.KeyGetTime(lCount)
        data = {
            "Key Time": lKeyTime.GetTimeString(lTimeString),
            "Key Value": pProperty.GetEnumValue(lKeyValue),
        }

        # lOutputString = "            Key Time: "
        # lOutputString += lKeyTime.GetTimeString(lTimeString)
        # lOutputString += ".... Key Value: "
        # lOutputString += lKeyValue
        # lOutputString += " ("
        # lOutputString += pProperty.GetEnumValue(lKeyValue)
        # lOutputString += ")"

        # print(lOutputString)
        ListCurveKeys.append(data)
    return ListCurveKeys


def DisplayListCurveDefault(pCurve, pProperty):
    GetCurveDefault(pCurve)
