from enum import Enum

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"

class ComputeType(Enum):
    INT8 = "int8"
    FLOAT16 = "float16"

class ModelSize(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"

class ComputeType(Enum):
    INT8 = "int8"
    FLOAT16 = "float16"