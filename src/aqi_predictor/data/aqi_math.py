"""
US EPA Air Quality Index (AQI) math.

Open-Meteo's Air Quality API already returns a correctly-computed `us_aqi`
value (it applies the proper rolling averages per pollutant internally), so
this module is *not* on the critical path when `AQI_DATA_PROVIDER=open_meteo`
(the default).

It exists to support the `openweather` provider, whose Air Pollution API only
returns raw pollutant concentrations (in ug/m3) plus OpenWeather's own 1-5
scale -- not the 0-500 US EPA index this project standardises on. We derive
an approximate US AQI from those concentrations here so both providers yield
a target variable on the same scale.

Reference: US EPA breakpoint table (2024 revision), as published by
Open-Meteo: https://open-meteo.com/en/docs/air-quality-api

Important limitation: the official US AQI uses rolling-window averages
(24h for PM2.5/PM10, 8h for CO/O3, 1h or 24h for SO2/NO2). A single API
snapshot only gives us instantaneous concentrations, so the value computed
here is an *approximation* using instantaneous readings against the same
breakpoints. This is clearly noted wherever it is used.
"""

from __future__ import annotations

from dataclasses import dataclass

# Molar masses (g/mol), used to convert ug/m3 -> ppb/ppm at 25 degC, 1 atm.
_MOLAR_MASS = {
    "co": 28.01,
    "no2": 46.0055,
    "so2": 64.066,
    "o3": 48.00,
}
_MOLAR_VOLUME_25C = 24.45  # L/mol at 25 degC and 1 atm


def ugm3_to_ppb(concentration_ugm3: float, pollutant: str) -> float:
    """Convert a ug/m3 concentration to ppb for a gas pollutant."""
    mw = _MOLAR_MASS[pollutant]
    return (concentration_ugm3 * _MOLAR_VOLUME_25C) / mw


def ugm3_to_ppm(concentration_ugm3: float, pollutant: str) -> float:
    """Convert a ug/m3 concentration to ppm for a gas pollutant."""
    return ugm3_to_ppb(concentration_ugm3, pollutant) / 1000.0


@dataclass(frozen=True)
class _Breakpoint:
    conc_low: float
    conc_high: float
    aqi_low: int
    aqi_high: int


# Each pollutant maps to an ordered list of (concentration range -> AQI range)
# breakpoints. Concentration units: PM2.5/PM10 in ug/m3, CO in ppm, SO2/NO2/O3
# in ppb.
_BREAKPOINTS: dict[str, list[_Breakpoint]] = {
    "pm2_5": [
        _Breakpoint(0.0, 9.0, 0, 50),
        _Breakpoint(9.1, 35.4, 51, 100),
        _Breakpoint(35.5, 55.4, 101, 150),
        _Breakpoint(55.5, 125.4, 151, 200),
        _Breakpoint(125.5, 225.4, 201, 300),
        _Breakpoint(225.5, 325.4, 301, 400),
        _Breakpoint(325.5, 500.4, 401, 500),
    ],
    "pm10": [
        _Breakpoint(0, 54, 0, 50),
        _Breakpoint(55, 154, 51, 100),
        _Breakpoint(155, 254, 101, 150),
        _Breakpoint(255, 354, 151, 200),
        _Breakpoint(355, 424, 201, 300),
        _Breakpoint(425, 504, 301, 400),
        _Breakpoint(505, 604, 401, 500),
    ],
    "co": [  # ppm, 8h breakpoints applied to an instantaneous reading
        _Breakpoint(0.0, 4.4, 0, 50),
        _Breakpoint(4.5, 9.4, 51, 100),
        _Breakpoint(9.5, 12.4, 101, 150),
        _Breakpoint(12.5, 15.4, 151, 200),
        _Breakpoint(15.5, 30.4, 201, 300),
        _Breakpoint(30.5, 40.4, 301, 400),
        _Breakpoint(40.5, 50.4, 401, 500),
    ],
    "so2": [  # ppb, 1h breakpoints applied to an instantaneous reading
        _Breakpoint(0, 35, 0, 50),
        _Breakpoint(36, 75, 51, 100),
        _Breakpoint(76, 185, 101, 150),
        _Breakpoint(186, 304, 151, 200),
        _Breakpoint(305, 604, 201, 300),
        _Breakpoint(605, 804, 301, 400),
        _Breakpoint(805, 1004, 401, 500),
    ],
    "no2": [  # ppb, 1h breakpoints
        _Breakpoint(0, 53, 0, 50),
        _Breakpoint(54, 100, 51, 100),
        _Breakpoint(101, 360, 101, 150),
        _Breakpoint(361, 649, 151, 200),
        _Breakpoint(650, 1249, 201, 300),
        _Breakpoint(1250, 1649, 301, 400),
        _Breakpoint(1650, 2049, 401, 500),
    ],
    "o3": [  # ppb, 8h breakpoints
        _Breakpoint(0, 54, 0, 50),
        _Breakpoint(55, 70, 51, 100),
        _Breakpoint(71, 85, 101, 150),
        _Breakpoint(86, 105, 151, 200),
        _Breakpoint(106, 200, 201, 300),
    ],
}


def _linear_aqi(concentration: float, bp: _Breakpoint) -> float:
    """EPA's piecewise-linear interpolation formula within one breakpoint band."""
    return ((bp.aqi_high - bp.aqi_low) / (bp.conc_high - bp.conc_low)) * (
        concentration - bp.conc_low
    ) + bp.aqi_low


def _sub_index(concentration: float, pollutant: str) -> float | None:
    """AQI sub-index for a single pollutant, or None if out of table range."""
    if concentration is None or concentration < 0:
        return None
    breakpoints = _BREAKPOINTS[pollutant]
    for bp in breakpoints:
        if bp.conc_low <= concentration <= bp.conc_high:
            return _linear_aqi(concentration, bp)
    # Above the top of the table: clamp to the worst band's formula so a
    # very high reading still returns "very hazardous" rather than None.
    top = breakpoints[-1]
    if concentration > top.conc_high:
        return _linear_aqi(concentration, top)
    return None


def us_aqi_from_components(
    pm2_5: float | None = None,
    pm10: float | None = None,
    co: float | None = None,
    no2: float | None = None,
    so2: float | None = None,
    o3: float | None = None,
    gases_in_ugm3: bool = True,
) -> float | None:
    """
    Approximate consolidated US AQI (0-500) from pollutant concentrations.

    PM2.5 and PM10 are expected in ug/m3. CO, NO2, SO2 and O3 default to
    ug/m3 (`gases_in_ugm3=True`, matching OpenWeather's Air Pollution API)
    and are converted to ppm/ppb internally; pass `gases_in_ugm3=False` if
    you already have them in ppm/ppb.

    Returns the maximum of all available sub-indices (the official US AQI
    is defined as the worst-case pollutant), or None if no pollutant had
    a usable reading.
    """
    sub_indices: list[float] = []

    if pm2_5 is not None:
        idx = _sub_index(pm2_5, "pm2_5")
        if idx is not None:
            sub_indices.append(idx)

    if pm10 is not None:
        idx = _sub_index(pm10, "pm10")
        if idx is not None:
            sub_indices.append(idx)

    if co is not None:
        co_ppm = ugm3_to_ppm(co, "co") if gases_in_ugm3 else co
        idx = _sub_index(co_ppm, "co")
        if idx is not None:
            sub_indices.append(idx)

    if no2 is not None:
        no2_ppb = ugm3_to_ppb(no2, "no2") if gases_in_ugm3 else no2
        idx = _sub_index(no2_ppb, "no2")
        if idx is not None:
            sub_indices.append(idx)

    if so2 is not None:
        so2_ppb = ugm3_to_ppb(so2, "so2") if gases_in_ugm3 else so2
        idx = _sub_index(so2_ppb, "so2")
        if idx is not None:
            sub_indices.append(idx)

    if o3 is not None:
        o3_ppb = ugm3_to_ppb(o3, "o3") if gases_in_ugm3 else o3
        idx = _sub_index(o3_ppb, "o3")
        if idx is not None:
            sub_indices.append(idx)

    if not sub_indices:
        return None
    return round(max(sub_indices), 1)
