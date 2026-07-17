from enum import Enum


class CostCurrency(str, Enum):
    EUR = 'EUR'
    USD = 'USD'


def estimated_cost(hourly_price: float, elapsed_seconds: float) -> float:
    return hourly_price * elapsed_seconds / 3600
