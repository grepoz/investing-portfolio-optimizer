from datetime import datetime


def get_static_wallet_value(initial_cash, monthly_contribution, contribution_periods):
    return initial_cash + monthly_contribution * contribution_periods


def months_between_dates(start_date: str, end_date: str) -> int:
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    return (end.year - start.year) * 12 + end.month - start.month

