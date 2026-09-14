import math


def calculate_residue(c0, dt50, days):
    """
    Estimate remaining pesticide residue using first-order decay.

    c0   = initial residue concentration (mg/kg)
    dt50 = half-life (days)
    days = days after application
    """

    k = math.log(2) / dt50
    residue = c0 * math.exp(-k * days)

    return residue


def calculate_safe_harvest_time(c0, dt50, mrl):
    """
    Calculate the estimated time required for residue
    to fall to or below the MRL.
    """

    if c0 <= mrl:
        return 0

    k = math.log(2) / dt50
    safe_days = math.log(c0 / mrl) / k

    return safe_days
