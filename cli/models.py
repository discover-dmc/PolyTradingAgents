from enum import Enum


class AnalystType(str, Enum):
    NEWS = "news"
    BASE_RATE = "base_rate"
    CROWD_FORECAST = "crowd_forecast"
    DATA = "data"
