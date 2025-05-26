import ctypes as ct
import sys

# Definition of RM_PositionData struct - should match esminiRMLib::RM_PositionData struct
class RM_PositionData(ct.Structure):
    _fields_ = [
        ("x", ct.c_float),
        ("y", ct.c_float),
        ("z", ct.c_float),
        ("h", ct.c_float),
        ("p", ct.c_float),
        ("r", ct.c_float),
        ("h_relative", ct.c_float),
        ("road_id", ct.c_int),
        ("junction_id", ct.c_int),  # -1 if not in a junction
        ("lane_id", ct.c_int),
        ("lane_offset", ct.c_float),
        ("s", ct.c_float),
    ]

class CoordProjector:
    def __init__(self, xodr_path):
        if sys.platform == "linux" or sys.platform == "linux2":
            self.rm = ct.CDLL("./esmini/bin/libesminiRMLib.so")
        elif sys.platform == "darwin":
            self.rm = ct.CDLL("./esmini/bin/libesminiRMLib.dylib")
        elif sys.platform == "win32":
            self.rm = ct.CDLL("./esmini/bin/esminiRMLib.dll")
        else:
            print("Unsupported platform: {}".format(sys.platform))
            sys.exit(-1)

        # Specify argument types to a few functions
        self.rm.RM_SetWorldPosition.argtypes = [ct.c_int, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float]
        self.rm.RM_SetLanePosition.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_float]

        # Initialize esmini RoadManger with given OpenDRIVE file
        self.odr = xodr_path
        if self.rm.RM_Init(self.odr.encode()) == -1:   # encode() converts string to pure byte array
            print("Failed to load OpenDRIVE file ", )
            sys.exit(-1)

        self.rm_pos = self.rm.RM_CreatePosition()  # create a position object, returns a handle
        self.rm_pos_data = RM_PositionData()  # object that will be passed and filled in with position info

    def coord_project(self, x, y):
        self.rm.RM_SetWorldPosition(self.rm_pos, x, y, 0.0, 0.0, 0.0, 0.0)
        self.rm.RM_GetPositionData(self.rm_pos, ct.byref(self.rm_pos_data))
        return (self.rm_pos_data.road_id, self.rm_pos_data.lane_id, self.rm_pos_data.lane_offset, self.rm_pos_data.s)