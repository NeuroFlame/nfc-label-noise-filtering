from typing import NamedTuple, Dict, Any, TypedDict
from .logger import NvFlareLogger

class LabelMetaData(TypedDict):
  name: str

class ComputationParamDTO(TypedDict):
  SamplingThreshold: float
  Iteration: int
  NTree: int
  LabelThreshold: int
  TypicalThreshold: float
  TruncationParameter: float
  LabelDefinition: Dict[str, LabelMetaData]
  LogLevel: str

class ConfigDTO(NamedTuple):
  data_path: str
  output_path: str
  cache_path: str
  computation_params: ComputationParamDTO
  cache_dict: Dict[str, Any]
  logger: NvFlareLogger
  site_name: str
