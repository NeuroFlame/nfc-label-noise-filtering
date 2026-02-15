import site
from typing import NamedTuple, Dict, Any, TypedDict
from enum import Enum

from .logger import NvFlareLogger

class ComputationParamDTO(TypedDict):
  Covariates: Dict[str, str]
  ReferenceColumns: Dict[str, str]
  ScicaTemplatePath: str
  MaskPath: str
  SubsampleNiftiImages: bool
  VoxelSize: int
  LogLovel: str

class ConfigDTO(NamedTuple):
  data_path: str
  output_path: str
  cache_path: str
  computation_params: ComputationParamDTO
  cache_dict: Dict[str, Any]
  logger: NvFlareLogger
  site_name: str
