try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")
import torch
import numpy as np
import sys

# constants and flags
USE_TRAINING_CLICKS = False
OBJECT_CLICK_COLOR = [0.2, 0.81, 0.2]  # colors between 0 and 1 for open3d
BACKGROUND_CLICK_COLOR = [0.81, 0.2, 0.2]  # colors between 0 and 1 for open3d
UNSELECTED_OBJECTS_COLOR = [0.4, 0.4, 0.4]
SELECTED_OBJECT_COLOR = [0.2, 0.81, 0.2]

obj_color = {
    1: [1, 211, 211],
    2: [233, 138, 0],
    3: [41, 207, 2],
    4: [244, 0, 128],
    5: [194, 193, 3],
    6: [121, 59, 50],
    7: [254, 180, 214],
    8: [239, 1, 51],
    9: [85, 85, 85],
    10: [229, 14, 241],
    11: [39, 39, 215],
    12: [137, 217, 221],
    13: [153, 39, 38],
    14: [120, 37, 217],
    15: [97, 99, 162],
    16: [220, 139, 218],
    17: [38, 131, 35],
    18: [39, 221, 112],
    19: [220, 33, 37],
    20: [221, 148, 118],
    21: [40, 33, 134],
    22: [215, 86, 157],
    23: [39, 215, 214],
    24: [127, 126, 31],
    25: [121, 214, 32],
    26: [130, 32, 136],
    27: [35, 159, 148],
    28: [159, 102, 219],
    29: [32, 39, 38],
    30: [94, 154, 95],
    31: [205, 163, 38],
    32: [105, 220, 153],
    33: [153, 211, 93],
    34: [216, 217, 213],
    35: [92, 47, 51],
    36: [220, 99, 44],
}

# Numpy reader format
valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}


def get_obj_color(obj_idx, normalize=False):
    if type(obj_idx) == "str":
        obj_idx = int(obj_idx)
    r, g, b = obj_color[obj_idx]

    if normalize:
        r /= 256
        g /= 256
        b /= 256

    return [r, g, b]


def find_nearest(coordinates, value):
    distance = torch.cdist(coordinates, torch.tensor([value]).to(coordinates.device), p=2)
    return distance.argmin().tolist()


def mean_iou_single(pred, labels):
    truepositive = pred * labels
    intersection = torch.sum(truepositive == 1)
    uni = torch.sum(pred == 1) + torch.sum(labels == 1) - intersection

    iou = intersection / uni
    return iou


def mean_iou_scene(pred, labels):

    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids != 0]
    obj_num = len(obj_ids)
    iou_sample = 0.0
    iou_dict = {}
    for obj_id in obj_ids:
        obj_iou = mean_iou_single(pred == obj_id, labels == obj_id)
        iou_dict[int(obj_id)] = float(obj_iou)
        iou_sample += obj_iou

    iou_sample /= obj_num

    return iou_sample, iou_dict


###############################################################
################ PLY file helper functions ####################
###############################################################

# Define PLY types
ply_dtypes = dict([(b"int8", "i1"), (b"char", "i1"), (b"uint8", "u1"), (b"uchar", "u1"), (b"int16", "i2"), (b"short", "i2"), (b"uint16", "u2"), (b"ushort", "u2"), (b"int32", "i4"), (b"int", "i4"), (b"uint32", "u4"), (b"uint", "u4"), (b"float32", "f4"), (b"float", "f4"), (b"float64", "f8"), (b"double", "f8")])
# Numpy reader format
valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        # Find point element
        if b"element vertex" in line:
            current_element = "vertex"
            line = line.split()
            num_points = int(line[2])

        elif b"element face" in line:
            current_element = "face"
            line = line.split()
            num_faces = int(line[2])

        elif b"property" in line:
            if current_element == "vertex":
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == "vertex":
                if not line.startswith("property list uchar int"):
                    raise ValueError("Unsupported faces property : " + line)

    return num_points, num_faces, vertex_properties


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append("element vertex %d" % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append("property %s %s" % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, "rb") as plyfile:

        # Check if the file start with ply
        if b"ply" not in plyfile.readline():
            raise ValueError("The file does not start whith the word ply")

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [("k", ext + "u1"), ("v1", ext + "i4"), ("v2", ext + "i4"), ("v3", ext + "i4")]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data["v1"], faces_data["v2"], faces_data["v3"])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.
    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.
    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.
    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)
    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print("fields have more than 2 dimensions")
            return False

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print("wrong field dimensions")
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        print("wrong number of field names")
        return False

    # Add extension if not there
    if not filename.endswith(".ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as plyfile:

        # First magical word
        header = ["ply"]

        # Encoding format
        header.append("format binary_" + sys.byteorder + "_endian 1.0")

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append("element face {:d}".format(triangular_faces.shape[0]))
            header.append("property list uchar int vertex_indices")

        # End of header
        header.append("end_header")

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, "ab") as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [("k", "uint8")] + [(str(ind), "int32") for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data["k"] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data["0"] = triangular_faces[:, 0]
            data["1"] = triangular_faces[:, 1]
            data["2"] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True
