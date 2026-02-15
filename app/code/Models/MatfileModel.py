from enum import Enum

class SourceDataKeys(Enum):
    """
    Enum to represent different keys in the original mat file.
    """
    FILE_ID = 'FILE_ID'
    ANALYSIS_ID = 'analysis_ID'
    ANALYSIS_SCORE = 'analysis_SCORE'
    SFNC = 'sFNC'
