from collections import namedtuple
import numpy as np

###############################
#### SemanticKITTI dataset ####
###############################

semantickitti_label_mapping = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}
semantickitti_ignore_labels = {"unlabeled"}
semantickitti_thing_labels = {"car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist"}
semantickitti_stuff_labels = {"road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"}

###############################
####### nuScenes dataset ######
###############################
# general
nuScenes_general_label_mapping = {
    0: "noise",
    1: "animal",
    2: "human.pedestrian.adult",
    3: "human.pedestrian.child",
    4: "human.pedestrian.construction_worker",
    5: "human.pedestrian.personal_mobility",
    6: "human.pedestrian.police_officer",
    7: "human.pedestrian.stroller",
    8: "human.pedestrian.wheelchair",
    9: "movable_object.barrier",
    10: "movable_object.debris",
    11: "movable_object.pushable_pullable",
    12: "movable_object.trafficcone",
    13: "static_object.bicycle_rack",
    14: "vehicle.bicycle",
    15: "vehicle.bus.bendy",
    16: "vehicle.bus.rigid",
    17: "vehicle.car",
    18: "vehicle.construction",
    19: "vehicle.emergency.ambulance",
    20: "vehicle.emergency.police",
    21: "vehicle.motorcycle",
    22: "vehicle.trailer",
    23: "vehicle.truck",
    24: "flat.driveable_surface",
    25: "flat.other",
    26: "flat.sidewalk",
    27: "flat.terrain",
    28: "static.manmade",
    29: "static.other",
    30: "static.vegetation",
    31: "vehicle.ego",
}
nuScenes_general_ignore_labels = {"noise"}
nuScenes_general_thing_labels = {
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
}
nuScenes_general_stuff_labels = {"flat.driveable_surface", "flat.other", "flat.sidewalk", "flat.terrain", "static.manmade", "static.other", "static.vegetation", "vehicle.ego"}
# challenge
nuScenes_challenge_label_mapping = {
    0: "void / ignore",
    1: "barrier (thing)",
    2: "bicycle (thing)",
    3: "bus (thing)",
    4: "car (thing)",
    5: "construction_vehicle (thing)",
    6: "motorcycle (thing)",
    7: "pedestrian (thing)",
    8: "traffic_cone (thing)",
    9: "trailer (thing)",
    10: "truck (thing)",
    11: "driveable_surface (stuff)",
    12: "other_flat (stuff)",
    13: "sidewalk (stuff)",
    14: "terrain (stuff)",
    15: "manmade (stuff)",
    16: "vegetation (stuff)",
}
nuScenes_challenge_ignore_labels = {"void / ignore"}
nuScenes_challenge_thing_labels = {"barrier (thing)", "bicycle (thing)", "bus (thing)", "car (thing)", "construction_vehicle (thing)", "motorcycle (thing)", "pedestrian (thing)", "traffic_cone (thing)", "trailer (thing)", "truck (thing)"}
nuScenes_challenge_stuff_labels = {"driveable_surface (stuff)", "other_flat (stuff)", "sidewalk (stuff)", "terrain (stuff)", "manmade (stuff)", "vegetation (stuff)"}

###############################
####### KITTI360 dataset ######
###############################

Label = namedtuple("Label", ["name", "id", "kittiId", "trainId", "category", "categoryId", "hasInstances", "ignoreInEval", "ignoreInInst", "color"])
labels_info = [
    # name. id. kittiId, trainId   category, catId, hasInstances, ignoreInEval, ignoreInInst, color
    Label("unlabeled", 0, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("ego vehicle", 1, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("rectification border", 2, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("out of roi", 3, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("static", 4, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("dynamic", 5, -1, 255, "void", 0, False, True, True, (111, 74, 0)),
    Label("ground", 6, -1, 255, "void", 0, False, True, True, (81, 0, 81)),
    Label("road", 7, 1, 0, "flat", 1, False, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 3, 1, "flat", 1, False, False, False, (244, 35, 232)),
    Label("parking", 9, 2, 255, "flat", 1, False, True, True, (250, 170, 160)),
    Label("rail track", 10, 10, 255, "flat", 1, False, True, True, (230, 150, 140)),
    Label("building", 11, 11, 2, "construction", 2, True, False, False, (70, 70, 70)),
    Label("wall", 12, 7, 3, "construction", 2, False, False, False, (102, 102, 156)),
    Label("fence", 13, 8, 4, "construction", 2, False, False, False, (190, 153, 153)),
    Label("guard rail", 14, 30, 255, "construction", 2, False, True, True, (180, 165, 180)),
    Label("bridge", 15, 31, 255, "construction", 2, False, True, True, (150, 100, 100)),
    Label("tunnel", 16, 32, 255, "construction", 2, False, True, True, (150, 120, 90)),
    Label("pole", 17, 21, 5, "object", 3, True, False, True, (153, 153, 153)),
    Label("polegroup", 18, -1, 255, "object", 3, False, True, True, (153, 153, 153)),
    Label("traffic light", 19, 23, 6, "object", 3, True, False, True, (250, 170, 30)),
    Label("traffic sign", 20, 24, 7, "object", 3, True, False, True, (220, 220, 0)),
    Label("vegetation", 21, 5, 8, "nature", 4, False, False, False, (107, 142, 35)),
    Label("terrain", 22, 4, 9, "nature", 4, False, False, False, (152, 251, 152)),
    Label("sky", 23, 9, 10, "sky", 5, False, False, False, (70, 130, 180)),
    Label("person", 24, 19, 11, "human", 6, True, False, False, (220, 20, 60)),
    Label("rider", 25, 20, 12, "human", 6, True, False, False, (255, 0, 0)),
    Label("car", 26, 13, 13, "vehicle", 7, True, False, False, (0, 0, 142)),
    Label("truck", 27, 14, 14, "vehicle", 7, True, False, False, (0, 0, 70)),
    Label("bus", 28, 34, 15, "vehicle", 7, True, False, False, (0, 60, 100)),
    Label("caravan", 29, 16, 255, "vehicle", 7, True, True, True, (0, 0, 90)),
    Label("trailer", 30, 15, 255, "vehicle", 7, True, True, True, (0, 0, 110)),
    Label("train", 31, 33, 16, "vehicle", 7, True, False, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, 17, "vehicle", 7, True, False, False, (0, 0, 230)),
    Label("bicycle", 33, 18, 18, "vehicle", 7, True, False, False, (119, 11, 32)),
    Label("garage", 34, 12, 2, "construction", 2, True, True, True, (64, 128, 128)),
    Label("gate", 35, 6, 4, "construction", 2, False, True, True, (190, 153, 153)),
    Label("stop", 36, 29, 255, "construction", 2, True, True, True, (150, 120, 90)),
    Label("smallpole", 37, 22, 5, "object", 3, True, True, True, (153, 153, 153)),
    Label("lamp", 38, 25, 255, "object", 3, True, True, True, (0, 64, 64)),
    Label("trash bin", 39, 26, 255, "object", 3, True, True, True, (0, 128, 192)),
    Label("vending machine", 40, 27, 255, "object", 3, True, True, True, (128, 64, 0)),
    Label("box", 41, 28, 255, "object", 3, True, True, True, (64, 64, 128)),
    Label("unknown construction", 42, 35, 255, "void", 0, False, True, True, (102, 0, 0)),
    Label("unknown vehicle", 43, 36, 255, "void", 0, False, True, True, (51, 0, 51)),
    Label("unknown object", 44, 37, 255, "void", 0, False, True, True, (32, 32, 32)),
    Label("license plate", -1, -1, -1, "vehicle", 7, False, True, True, (0, 0, 142)),
]

label_name_mapping_kitti360 = {label.id: label.name for label in labels_info}
label2category = {label.id: label.category for label in labels_info}


def get_things_stuff_split_kitti360():
    thing_labels = set()
    stuff_labels = set()

    for label in labels_info:
        if label.category == "human" or label.category == "vehicle":
            thing_labels.add(label.name)
        else:
            stuff_labels.add(label.name)

    return thing_labels, stuff_labels
