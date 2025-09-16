from enums.config_enums import DeviceType, ModelSize, ComputeType

API_KEY = ""
RATE = 16000 # Audio sampling rate
CHANNELS = 1
CHUNK = 1024
FORMAT = 8  #
DURATION = 8 # length of sound(s)
FILE_BEAM_SIZE = 1
AUDIO_BEAM_SIZE = 3
MODEL_SIZE = ModelSize.BASE.value
EMBEDDING_MODEL = "models/gemini-embedding-001"
CONTENT_GEN_MODEL = "models/gemini-2.0-flash"
DEVICE = DeviceType.CPU.value
COMPUTE_TYPE = ComputeType.INT8.value