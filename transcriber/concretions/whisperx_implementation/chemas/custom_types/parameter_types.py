from enum import Enum

class ComputeType(str, Enum):
    float32 = "float32"
    float16 = "float16"
    int8 = "int8"
    int4 = "int4"
    int2 = "int2"
    int1 = "int1"
    int0 = "int0"


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"

class WhisperModel(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"
    large_v2 = "large-v2"
    large_v3 = "large-v3"