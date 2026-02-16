from typing import TypedDict
from typing_extensions import NotRequired


class HeatMapOptions(TypedDict):
    colorbar_name: str
    title: str
    name: str
    domain_names: NotRequired[list[int]]
    path: str
