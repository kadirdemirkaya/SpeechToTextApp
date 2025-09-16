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
    MEDIUM = "medium"
    LARGE1 = "large-v1"
    LARGE2 = "large-v2"

class ComputeType(Enum):
    INT8 = "int8"
    FLOAT16 = "float16"