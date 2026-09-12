from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import jwt
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pwdlib import PasswordHash

# ============================================================
# CONFIGURATION
# ============================================================

class Settings(BaseSettings):
    app_name: str = Field(default="WeatherGPT")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)

    weather_api_url: str = Field(
        default="https://api.open-meteo.com/v1/forecast"
    )
    weather_api_key: str | None = Field(default=None)

    llm_api_key: str | None = Field(default=None)
    llm_api_url: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions"
    )
    llm_model: str = Field(default="google/gemma-3-27b-it:free")

    jwt_secret: str = Field(default="development-secret-change-me")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)

    default_language: str = Field(default="en")
    default_timezone: str = Field(default="Asia/Kolkata")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()

# database.json is deliberately separate from this Python file.
DATABASE_FILE = Path(__file__).with_name("database.json")

# ============================================================
# JSON DATABASE
# ============================================================

DEFAULT_DATABASE = {
    "users": [],
    "locations": [],
    "user_preferences": [],
    "chat_messages": [],
    "weather_records": [],
    "forecasts": [],
    "alerts": [],
    "advisories": [],
    "next_ids": {
        "users": 1,
        "locations": 1,
        "user_preferences": 1,
        "chat_messages": 1,
        "weather_records": 1,
        "forecasts": 1,
        "alerts": 1,
        "advisories": 1,
    },
}


def load_database() -> dict[str, Any]:
    if not DATABASE_FILE.exists():
        save_database(DEFAULT_DATABASE.copy())

    try:
        with DATABASE_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("database.json could not be read.") from exc

    for key, value in DEFAULT_DATABASE.items():
        if key not in data:
            data[key] = value.copy() if isinstance(value, dict) else list(value)

    return data


def save_database(data: dict[str, Any]) -> None:
    temp_file = DATABASE_FILE.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temp_file.replace(DATABASE_FILE)


def next_id(data: dict[str, Any], table: str) -> int:
    value = data["next_ids"].get(table, 1)
    data["next_ids"][table] = value + 1
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    data = load_database()
    return next((u for u in data["users"] if u["id"] == user_id), None)


def get_user_by_email(email: str) -> dict[str, Any] | None:
    email = email.lower().strip()
    data = load_database()
    return next((u for u in data["users"] if u["email"].lower() == email), None)

# ============================================================
# SECURITY / JWT
# ============================================================

password_hash = PasswordHash.recommended()
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


def create_access_token(
    user_id: int | str,
    expires_delta: timedelta | None = None,
) -> str:
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_expire_minutes)

    expire = datetime.now(timezone.utc) + expires_delta
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    return user

# ============================================================
# USER SCHEMAS
# ============================================================

class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    language: str = Field(default="en", min_length=2, max_length=10)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UserPreferences(BaseModel):
    language: str = Field(default="en", min_length=2, max_length=10)
    timezone: str = Field(default="Asia/Kolkata", min_length=1, max_length=50)
    temperature_unit: str = Field(
        default="celsius",
        pattern="^(celsius|fahrenheit)$",
    )
    wind_speed_unit: str = Field(
        default="kmh",
        pattern="^(kmh|mph|ms)$",
    )
    notifications_enabled: bool = True
    severe_alerts_enabled: bool = True


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    language: str | None = Field(default=None, min_length=2, max_length=10)


class PasswordUpdate(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    language: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., gt=0)
    user: UserResponse

# ============================================================
# LOCATION SCHEMAS
# ============================================================

class LocationCreate(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    country_code: str | None = Field(default=None, min_length=2, max_length=2)


class LocationUpdate(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    country_code: str | None = Field(default=None, min_length=2, max_length=2)


class LocationResponse(BaseModel):
    id: int
    city: str
    latitude: float
    longitude: float
    country: str | None
    timezone: str | None

# ============================================================
# WEATHER SCHEMAS
# ============================================================

class LocationInfo(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    country: str | None = Field(default=None, max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timezone: str | None = Field(default=None, max_length=50)


class CurrentWeather(BaseModel):
    temperature: float
    feels_like: float | None = None
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)
    wind_direction: float | None = Field(default=None, ge=0, le=360)
    precipitation: float = Field(default=0.0, ge=0)
    pressure: float | None = Field(default=None, ge=0)
    weather_code: int | None = None
    description: str | None = Field(default=None, max_length=255)
    observed_at: datetime | None = None


class ForecastItem(BaseModel):
    forecast_time: datetime
    temperature: float
    feels_like: float | None = None
    humidity: float | None = Field(default=None, ge=0, le=100)
    precipitation_probability: float = Field(default=0.0, ge=0, le=100)
    precipitation: float = Field(default=0.0, ge=0)
    wind_speed: float = Field(default=0.0, ge=0)
    weather_code: int | None = None
    description: str | None = Field(default=None, max_length=255)


class CurrentWeatherResponse(BaseModel):
    location: LocationInfo
    current: CurrentWeather
    source: str
    fetched_at: datetime


class ForecastResponse(BaseModel):
    location: LocationInfo
    forecasts: list[ForecastItem]
    source: str
    model: str
    fetched_at: datetime


class WeatherAlert(BaseModel):
    alert_type: str
    severity: str
    title: str
    message: str
    value: float
    threshold: float
    unit: str
    source: str | None = None
    detected_at: str | None = None


class WeatherDashboardResponse(BaseModel):
    location: LocationInfo
    current: CurrentWeather
    forecast: list[ForecastItem]
    alerts: list[WeatherAlert] = Field(default_factory=list)
    fetched_at: datetime

# ============================================================
# CHAT SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=10)
    location: str | None = Field(default=None, max_length=100)
    conversation_id: str | None = Field(default=None, max_length=100)


class ChatLocation(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    country: str | None = Field(default=None, max_length=100)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class ChatResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    location: str | None = Field(default=None, max_length=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    language: str = Field(default="en", min_length=2, max_length=10)
    generated_at: datetime


class WeatherContext(BaseModel):
    temperature: float | None = None
    humidity: float | None = Field(default=None, ge=0, le=100)
    wind_speed: float | None = Field(default=None, ge=0)
    precipitation_probability: float | None = Field(default=None, ge=0, le=100)
    precipitation: float | None = Field(default=None, ge=0)
    weather_description: str | None = None
    observed_at: datetime | None = None


class ForecastSummary(BaseModel):
    date: str
    period: str
    timezone: str
    min_temperature: float | None = None
    max_temperature: float | None = None
    max_rain_probability: float | None = None
    total_precipitation: float | None = None


class DetailedChatResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    location: ChatLocation
    confidence: float = Field(..., ge=0, le=1)
    weather: WeatherContext
    forecast: ForecastSummary | None = None
    alerts: list[WeatherAlert] = Field(default_factory=list)
    language: str
    generated_at: datetime

# ============================================================
# SERVICE DATA CLASSES / EXCEPTIONS
# ============================================================

class LocationServiceError(Exception):
    pass


class LocationNotFoundError(LocationServiceError):
    pass


class LocationAPIError(LocationServiceError):
    pass


@dataclass
class LocationResult:
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    country_code: str | None = None
    state: str | None = None
    timezone: str | None = None


class WeatherServiceError(Exception):
    pass


class WeatherAPIError(WeatherServiceError):
    pass


class WeatherDataError(WeatherServiceError):
    pass


@dataclass
class CurrentWeatherData:
    temperature: float | None
    feels_like: float | None
    humidity: float | None
    wind_speed: float | None
    wind_direction: float | None
    precipitation: float | None
    pressure: float | None
    weather_code: int | None
    observed_at: str | None


@dataclass
class HourlyWeatherData:
    time: str
    temperature: float | None
    humidity: float | None
    precipitation_probability: float | None
    precipitation: float | None
    wind_speed: float | None
    weather_code: int | None


@dataclass
class DailyWeatherData:
    date: str
    temperature_max: float | None
    temperature_min: float | None
    precipitation_probability_max: float | None
    precipitation_sum: float | None
    wind_speed_max: float | None
    weather_code: int | None


@dataclass
class WeatherData:
    current: CurrentWeatherData
    hourly: list[HourlyWeatherData]
    daily: list[DailyWeatherData]
    source: str
    fetched_at: str
    timezone: str

# ============================================================
# LOCATION SERVICE
# ============================================================

CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
}

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
LOCATION_TIMEOUT = 10.0


def _validate_coordinates(latitude: float, longitude: float) -> bool:
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


async def _request_geocoding(
    city: str,
    country_code: str | None = None,
) -> dict:
    params: dict[str, Any] = {
        "name": city,
        "count": 5,
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["countryCode"] = country_code.upper()

    try:
        async with httpx.AsyncClient(timeout=LOCATION_TIMEOUT) as client:
            response = await client.get(GEOCODING_API_URL, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException as exc:
        print(f"WEATHER ERROR - TIMEOUT: {repr(exc)}")
        raise WeatherAPIError(
            "Weather API request timed out."
        ) from exc

    except httpx.HTTPStatusError as exc:
        print(
            f"WEATHER ERROR - HTTP {exc.response.status_code}: "
            f"{exc.response.text}"
        )
        raise WeatherAPIError(
            f"Weather API returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        print(f"WEATHER ERROR - REQUEST: {repr(exc)}")
        raise WeatherAPIError(
            "Unable to connect to the weather API."
        ) from exc

    except ValueError as exc:
        print(f"WEATHER ERROR - JSON: {repr(exc)}")
        raise WeatherAPIError(
            "Weather API returned invalid JSON."
        ) from exc


def _parse_location_result(data: dict, requested_city: str) -> LocationResult:
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise LocationNotFoundError(
            f"Location '{requested_city}' was not found."
        )

    # Prefer the first valid result. Open-Meteo already ranks results.
    for item in results:
        try:
            name = str(item["name"])
            latitude = float(item["latitude"])
            longitude = float(item["longitude"])
        except (KeyError, TypeError, ValueError):
            continue

        if not _validate_coordinates(latitude, longitude):
            continue

        return LocationResult(
            name=name,
            latitude=latitude,
            longitude=longitude,
            country=item.get("country"),
            country_code=item.get("country_code"),
            state=item.get("admin1"),
            timezone=item.get("timezone"),
        )

    raise LocationNotFoundError(
        f"Location '{requested_city}' was not found."
    )


async def geocode_city(
    city: str,
    country_code: str | None = None,
) -> LocationResult:
    city = city.strip()
    if not city:
        raise ValueError("City name cannot be empty.")

    normalized_city = CITY_ALIASES.get(city.lower(), city)
    data = await _request_geocoding(normalized_city, country_code)
    return _parse_location_result(data, city)


async def get_coordinates(city: str) -> tuple[float, float]:
    result = await geocode_city(city)
    return result.latitude, result.longitude

# ============================================================
# WEATHER SERVICE
# ============================================================

WEATHER_TIMEOUT = 20.0
DEFAULT_FORECAST_DAYS = 7


async def get_weather(
    latitude: float,
    longitude: float,
    forecast_days: int = DEFAULT_FORECAST_DAYS,
) -> WeatherData:
    if not _validate_coordinates(latitude, longitude):
        raise ValueError("Invalid geographic coordinates.")
    if not 1 <= forecast_days <= 7:
        raise ValueError("forecast_days must be between 1 and 7.")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,"
            "weather_code"
        ),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation_probability,"
            "precipitation,wind_speed_10m,weather_code"
        ),
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,precipitation_sum,"
            "wind_speed_10m_max,weather_code"
        ),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=WEATHER_TIMEOUT) as client:
            response = await client.get(settings.weather_api_url, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        print(f"WEATHER ERROR - TIMEOUT: {exc}")
        raise WeatherAPIError("Weather service timed out.") from exc

    except httpx.HTTPStatusError as exc:
        print(
            f"WEATHER ERROR - HTTP {exc.response.status_code}: "
            f"{exc.response.text}"
        )
        raise WeatherAPIError(
            f"Weather service returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        print(f"WEATHER ERROR - REQUEST: {repr(exc)}")
        raise WeatherAPIError(
            "Unable to connect to the weather service."
        ) from exc

    except ValueError as exc:
        print(f"WEATHER ERROR - JSON: {repr(exc)}")
        raise WeatherAPIError(
            "Weather service returned invalid JSON."
        ) from exc

    try:
        current = data["current"]
        hourly = data["hourly"]
        daily = data["daily"]
        timezone_name = data.get("timezone", "UTC")

        current_data = CurrentWeatherData(
            temperature=current.get("temperature_2m"),
            feels_like=current.get("apparent_temperature"),
            humidity=current.get("relative_humidity_2m"),
            wind_speed=current.get("wind_speed_10m"),
            wind_direction=current.get("wind_direction_10m"),
            precipitation=current.get("precipitation", 0.0),
            pressure=current.get("pressure_msl"),
            weather_code=current.get("weather_code"),
            observed_at=current.get("time"),
        )

        hourly_times = hourly.get("time", [])
        hourly_temperature = hourly.get("temperature_2m", [])
        hourly_humidity = hourly.get("relative_humidity_2m", [])
        hourly_rain_probability = hourly.get("precipitation_probability", [])
        hourly_precipitation = hourly.get("precipitation", [])
        hourly_wind = hourly.get("wind_speed_10m", [])
        hourly_code = hourly.get("weather_code", [])

        hourly_data = []
        for i, time_value in enumerate(hourly_times):
            hourly_data.append(
                HourlyWeatherData(
                    time=time_value,
                    temperature=hourly_temperature[i] if i < len(hourly_temperature) else None,
                    humidity=hourly_humidity[i] if i < len(hourly_humidity) else None,
                    precipitation_probability=(
                        hourly_rain_probability[i]
                        if i < len(hourly_rain_probability)
                        else None
                    ),
                    precipitation=(
                        hourly_precipitation[i]
                        if i < len(hourly_precipitation)
                        else None
                    ),
                    wind_speed=hourly_wind[i] if i < len(hourly_wind) else None,
                    weather_code=hourly_code[i] if i < len(hourly_code) else None,
                )
            )

        daily_times = daily.get("time", [])
        daily_max = daily.get("temperature_2m_max", [])
        daily_min = daily.get("temperature_2m_min", [])
        daily_probability = daily.get("precipitation_probability_max", [])
        daily_precipitation = daily.get("precipitation_sum", [])
        daily_wind = daily.get("wind_speed_10m_max", [])
        daily_code = daily.get("weather_code", [])

        daily_data = []
        for i, date_value in enumerate(daily_times):
            daily_data.append(
                DailyWeatherData(
                    date=date_value,
                    temperature_max=daily_max[i] if i < len(daily_max) else None,
                    temperature_min=daily_min[i] if i < len(daily_min) else None,
                    precipitation_probability_max=(
                        daily_probability[i] if i < len(daily_probability) else None
                    ),
                    precipitation_sum=(
                        daily_precipitation[i]
                        if i < len(daily_precipitation)
                        else None
                    ),
                    wind_speed_max=(
                        daily_wind[i] if i < len(daily_wind) else None
                    ),
                    weather_code=daily_code[i] if i < len(daily_code) else None,
                )
            )
    except (KeyError, TypeError, IndexError) as exc:
        raise WeatherDataError(
            "Weather provider returned invalid or incomplete data."
        ) from exc

    return WeatherData(
        current=current_data,
        hourly=hourly_data,
        daily=daily_data,
        source="Open-Meteo",
        fetched_at=utc_now(),
        timezone=timezone_name,
    )

# ============================================================
# WEATHER PRESENTATION HELPERS
# ============================================================

WEATHER_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _weather_description(weather_code: int | None) -> str | None:
    if weather_code is None:
        return None
    return WEATHER_DESCRIPTIONS.get(
        weather_code,
        "Unknown weather conditions",
    )


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return datetime.now(timezone.utc)

# ============================================================
# ALERT SERVICE
# ============================================================

@dataclass
class AlertResult:
    alert_type: str
    severity: str
    title: str
    message: str
    value: float
    threshold: float
    unit: str
    source: str | None = "WeatherGPT rules"
    detected_at: str | None = None


def evaluate_weather_alerts(weather_data: WeatherData) -> list[AlertResult]:
    """
    Rule-based extreme-weather demo alerts.

    The uploaded route files call an existing alert_service.py, but that
    implementation was not included in the supplied files. These rules are
    therefore a compatible replacement, not a claim that they are byte-for-byte
    identical to the missing service.
    """
    alerts: list[AlertResult] = []
    current = weather_data.current
    detected_at = current.observed_at or utc_now()

    # Heat warning.
    if current.temperature is not None and current.temperature >= 40:
        severity = "severe" if current.temperature >= 45 else "high"
        alerts.append(
            AlertResult(
                alert_type="heatwave",
                severity=severity,
                title="High temperature warning",
                message=(
                    f"Temperature is {current.temperature:.1f} °C. "
                    "Take precautions against heat exposure."
                ),
                value=float(current.temperature),
                threshold=40.0,
                unit="°C",
                detected_at=detected_at,
            )
        )

    # Strong wind warning.
    if current.wind_speed is not None and current.wind_speed >= 50:
        severity = "severe" if current.wind_speed >= 75 else "high"
        alerts.append(
            AlertResult(
                alert_type="strong_wind",
                severity=severity,
                title="Strong wind warning",
                message=(
                    f"Wind speed is {current.wind_speed:.1f} km/h. "
                    "Secure loose outdoor objects and take care."
                ),
                value=float(current.wind_speed),
                threshold=50.0,
                unit="km/h",
                detected_at=detected_at,
            )
        )

    # Thunderstorm warning from WMO codes.
    if current.weather_code in {95, 96, 99}:
        severity = "severe" if current.weather_code in {96, 99} else "high"
        alerts.append(
            AlertResult(
                alert_type="thunderstorm",
                severity=severity,
                title="Thunderstorm warning",
                message=(
                    f"Current conditions indicate "
                    f"{_weather_description(current.weather_code).lower()}."
                ),
                value=float(current.weather_code),
                threshold=95.0,
                unit="WMO code",
                detected_at=detected_at,
            )
        )

    return alerts

# ============================================================
# FORECAST SERVICE
# ============================================================

class ForecastServiceError(Exception):
    pass


class ForecastNotFoundError(ForecastServiceError):
    pass


class ForecastTimeError(ForecastServiceError):
    pass


@dataclass
class ForecastSelection:
    period: str
    target_date: date
    items: list[HourlyWeatherData]
    timezone: str


def _get_timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ForecastTimeError(
            f"Invalid timezone: {timezone_name}"
        ) from exc


def _get_local_today(timezone_name: str) -> date:
    return datetime.now(_get_timezone(timezone_name)).date()


def _get_forecast_datetime(item: HourlyWeatherData) -> datetime | None:
    try:
        return datetime.fromisoformat(item.time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _normalize_time_period(time_period: str | None) -> str:
    if not time_period:
        return "today"

    value = time_period.strip().lower()
    aliases = {
        "today": "today",
        "today's": "today",
        "tomorrow": "tomorrow",
        "tomorrow's": "tomorrow",
        "next day": "tomorrow",
        "morning": "today morning",
        "this morning": "today morning",
        "today morning": "today morning",
        "afternoon": "today afternoon",
        "this afternoon": "today afternoon",
        "today afternoon": "today afternoon",
        "evening": "today evening",
        "this evening": "today evening",
        "today evening": "today evening",
        "tonight": "today night",
        "night": "today night",
        "today night": "today night",
        "now": "now",
        "current": "now",
        "right now": "now",
        "tomorrow morning": "tomorrow morning",
        "tomorrow afternoon": "tomorrow afternoon",
        "tomorrow evening": "tomorrow evening",
        "tomorrow night": "tomorrow night",
        "this weekend": "this weekend",
    }
    return aliases.get(value, value)


def _filter_by_date(items: list[HourlyWeatherData], target_date: date) -> list[HourlyWeatherData]:
    selected = []
    for item in items:
        forecast_datetime = _get_forecast_datetime(item)
        if forecast_datetime and forecast_datetime.date() == target_date:
            selected.append(item)
    return selected


def _filter_by_date_and_time(
    items: list[HourlyWeatherData],
    target_date: date,
    start_hour: int,
    end_hour: int,
) -> list[HourlyWeatherData]:
    selected = []
    for item in items:
        forecast_datetime = _get_forecast_datetime(item)
        if forecast_datetime is None:
            continue
        if (
            forecast_datetime.date() == target_date
            and start_hour <= forecast_datetime.hour < end_hour
        ):
            selected.append(item)
    return selected


def select_forecast(
    weather_data: WeatherData,
    time_period: str | None,
    timezone_name: str,
) -> ForecastSelection:
    period = _normalize_time_period(time_period)
    local_today = _get_local_today(timezone_name)

    if period == "now":
        now = datetime.now(_get_timezone(timezone_name))
        candidates = []
        for item in weather_data.hourly:
            dt = _get_forecast_datetime(item)
            if dt is not None:
                candidates.append((abs((dt - now).total_seconds()), item))
        if not candidates:
            raise ForecastNotFoundError("No current forecast data is available.")
        closest = min(candidates, key=lambda pair: pair[0])[1]
        return ForecastSelection(
            period="now",
            target_date=local_today,
            items=[closest],
            timezone=timezone_name,
        )

    if period == "today":
        target_date = local_today
        items = _filter_by_date(weather_data.hourly, target_date)
    elif period == "tomorrow":
        target_date = local_today + timedelta(days=1)
        items = _filter_by_date(weather_data.hourly, target_date)
    elif period == "today morning":
        target_date = local_today
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 5, 12)
    elif period == "today afternoon":
        target_date = local_today
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 12, 17)
    elif period == "today evening":
        target_date = local_today
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 17, 21)
    elif period == "today night":
        target_date = local_today
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 21, 24)
    elif period == "tomorrow morning":
        target_date = local_today + timedelta(days=1)
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 5, 12)
    elif period == "tomorrow afternoon":
        target_date = local_today + timedelta(days=1)
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 12, 17)
    elif period == "tomorrow evening":
        target_date = local_today + timedelta(days=1)
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 17, 21)
    elif period == "tomorrow night":
        target_date = local_today + timedelta(days=1)
        items = _filter_by_date_and_time(weather_data.hourly, target_date, 21, 24)
    elif period == "this weekend":
        days_until_saturday = (5 - local_today.weekday()) % 7
        saturday = local_today + timedelta(days=days_until_saturday)
        sunday = saturday + timedelta(days=1)
        items = [
            item
            for item in weather_data.hourly
            if (_get_forecast_datetime(item) is not None)
            and _get_forecast_datetime(item).date() in {saturday, sunday}
        ]
        target_date = saturday
    else:
        # Preserve compatibility with the original service while giving a
        # deterministic error for unsupported periods.
        raise ForecastNotFoundError(
            f"Forecast period '{period}' is not available."
        )

    if not items:
        raise ForecastNotFoundError(
            f"No forecast data is available for '{period}'."
        )

    return ForecastSelection(
        period=period,
        target_date=target_date,
        items=items,
        timezone=timezone_name,
    )


def forecast_items_to_dict(selection: ForecastSelection) -> list[dict[str, Any]]:
    result = []
    for item in selection.items:
        result.append(
            {
                "forecast_time": item.time,
                "temperature": item.temperature,
                "humidity": item.humidity,
                "precipitation_probability": item.precipitation_probability,
                "precipitation": item.precipitation,
                "wind_speed": item.wind_speed,
                "weather_code": item.weather_code,
                "description": _weather_description(item.weather_code),
            }
        )
    return result

# ============================================================
# AI SERVICE
# ============================================================

class AIServiceError(Exception):
    pass


class AIConfigurationError(AIServiceError):
    pass


class AIAPIError(AIServiceError):
    pass


class AIResponseError(AIServiceError):
    pass


@dataclass
class WeatherIntent:
    intent: str
    location: str | None
    time_period: str | None
    language: str


@dataclass
class AIAnswer:
    answer: str
    language: str
    confidence: float
    intent: str


REQUEST_TIMEOUT = 30.0
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"


def _get_api_key() -> str:
    if not settings.llm_api_key:
        raise AIConfigurationError("LLM API key is not configured.")
    return settings.llm_api_key


def _get_model() -> str:
    return getattr(settings, "llm_model", None) or DEFAULT_MODEL


def _get_api_url() -> str:
    return (
        getattr(settings, "llm_api_url", None)
        or "https://openrouter.ai/api/v1/chat/completions"
    )


async def _call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
) -> str:
    payload = {
        "model": _get_model(),
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                _get_api_url(),
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise AIAPIError("LLM request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise AIAPIError(
            f"LLM API returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise AIAPIError(
            "Unable to connect to the LLM service."
        ) from exc
    except ValueError as exc:
        raise AIAPIError("LLM API returned invalid JSON.") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AIResponseError(
            "LLM returned an unexpected response format."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise AIResponseError("LLM returned an empty response.")

    return content.strip()


async def detect_language(message: str) -> str:
    system_prompt = """
You are a language detection component for a weather assistant.
Identify the primary language of the user's message.
Return ONLY a two-letter ISO-style language code.
Examples:
English -> en
Hindi -> hi
Marathi -> mr
Gujarati -> gu
Tamil -> ta
Telugu -> te
Bengali -> bn
Kannada -> kn
Malayalam -> ml
Punjabi -> pa
If uncertain, return en.
""".strip()

    result = await _call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message.strip()},
        ],
        temperature=0.0,
    )
    language = result.lower().strip().split()[0]
    return language if len(language) == 2 else "en"


async def extract_weather_intent(
    message: str,
    language: str = "en",
) -> WeatherIntent:
    system_prompt = """
You are the intent extraction component of WeatherGPT.
Extract ONLY the information needed by the backend.
Return EXACTLY four lines:
intent: <intent>
location: <location or none>
time_period: <time period or none>
language: <language code>

Allowed intents:
current_weather
hourly_forecast
daily_forecast
rain_forecast
temperature
wind
humidity
precipitation
weather_alert
general_weather
unknown

Preserve complete time expressions such as:
today
tomorrow
today morning
today afternoon
today evening
today night
tomorrow morning
tomorrow afternoon
tomorrow evening
tomorrow night
this weekend

Rules:
- current/now/right now -> now
- morning/afternoon/evening/tonight without a specific future day refer to today
- tomorrow morning must remain tomorrow morning
- tomorrow afternoon must remain tomorrow afternoon
- tomorrow evening must remain tomorrow evening
- tomorrow night must remain tomorrow night
- extract the city/location exactly as reasonably stated
- if no location is present, return location: none
""".strip()

    result = await _call_llm(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Preferred language: {language}\n\n"
                    f"User question:\n{message}"
                ),
            },
        ],
        temperature=0.0,
    )
    return _parse_intent_response(result)


def _parse_intent_response(response: str) -> WeatherIntent:
    values: dict[str, str] = {}
    for line in response.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()

    allowed_intents = {
        "current_weather",
        "hourly_forecast",
        "daily_forecast",
        "rain_forecast",
        "temperature",
        "wind",
        "humidity",
        "precipitation",
        "weather_alert",
        "general_weather",
        "unknown",
    }

    intent = values.get("intent", "unknown").lower()
    if intent not in allowed_intents:
        intent = "unknown"

    location = values.get("location")
    if location and location.lower() in {"none", "unknown", "null"}:
        location = None

    time_period = values.get("time_period")
    if time_period and time_period.lower() in {"none", "unknown", "null"}:
        time_period = None

    language = values.get("language", "en").lower()
    if len(language) != 2:
        language = "en"

    return WeatherIntent(
        intent=intent,
        location=location,
        time_period=time_period,
        language=language,
    )


def _format_weather_data(weather_data: dict[str, Any]) -> str:
    lines: list[str] = []
    location = weather_data.get("location", {})
    if isinstance(location, dict):
        lines.append(f"Location: {location.get('city', 'Unknown')}")
        if location.get("country"):
            lines.append(f"Country: {location['country']}")

    current = weather_data.get("current", {})
    if isinstance(current, dict):
        lines.append("")
        lines.append("CURRENT WEATHER:")
        fields = {
            "Temperature": ("temperature", "°C"),
            "Feels like": ("feels_like", "°C"),
            "Humidity": ("humidity", "%"),
            "Wind speed": ("wind_speed", "km/h"),
            "Wind direction": ("wind_direction", "°"),
            "Precipitation": ("precipitation", "mm"),
            "Pressure": ("pressure", "hPa"),
            "Weather description": ("description", ""),
        }
        for label, (key, unit) in fields.items():
            value = current.get(key)
            if value is not None:
                lines.append(f"{label}: {value}{(' ' + unit) if unit else ''}")

    forecast = weather_data.get("forecast", {})
    if isinstance(forecast, dict):
        lines.append("")
        lines.append("SELECTED FORECAST:")
        if forecast.get("target_date"):
            lines.append(f"Forecast date: {forecast['target_date']}")
        if forecast.get("period"):
            lines.append(f"Forecast period: {forecast['period']}")
        if forecast.get("timezone"):
            lines.append(f"Forecast timezone: {forecast['timezone']}")
        lines.append("")
        for item in forecast.get("items", []):
            if not isinstance(item, dict):
                continue
            lines.append(f"Time: {item.get('forecast_time', 'Unknown')}")
            if item.get("temperature") is not None:
                lines.append(f"Temperature: {item['temperature']} °C")
            if item.get("precipitation_probability") is not None:
                lines.append(
                    f"Rain probability: {item['precipitation_probability']}%"
                )
            if item.get("precipitation") is not None:
                lines.append(f"Precipitation: {item['precipitation']} mm")

    alerts = weather_data.get("alerts", [])
    if isinstance(alerts, list):
        lines.append("")
        lines.append("WEATHER ALERTS:")
        if not alerts:
            lines.append("No weather alerts supplied.")
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            lines.append(f"Severity: {alert.get('severity', 'unknown')}")
            lines.append(f"Title: {alert.get('title', 'Unknown')}")
            lines.append(f"Message: {alert.get('message', '')}")

    return "\n".join(lines)


def _calculate_answer_confidence(weather_data: dict[str, Any]) -> float:
    score = 0.0
    current = weather_data.get("current")
    if isinstance(current, dict):
        if current.get("temperature") is not None:
            score += 0.20
        if current.get("humidity") is not None:
            score += 0.15
        if current.get("wind_speed") is not None:
            score += 0.15
        if current.get("precipitation") is not None:
            score += 0.15
        if current.get("description") is not None:
            score += 0.10

    forecast = weather_data.get("forecast")
    if isinstance(forecast, dict) and forecast.get("items"):
        score += 0.25

    return round(min(score, 1.0), 2)


async def generate_weather_answer(
    user_message: str,
    weather_data: dict[str, Any],
    language: str = "en",
    intent: str = "general_weather",
) -> AIAnswer:
    system_prompt = """
You are WeatherGPT, a conversational weather assistant.

Your ONLY task is to answer the user's weather question using the VERIFIED WEATHER DATA supplied below.

IMPORTANT:
1. Answer the USER'S QUESTION directly.
2. Use ONLY the supplied weather data for weather facts.
3. NEVER invent temperature, rain probability, humidity, wind speed, precipitation, alerts, or forecast information.
4. If a requested weather value is unavailable, say that the information is unavailable.
5. Never make up weather alerts.
6. A rain probability is a probability, NOT a guarantee.
7. Respect the selected forecast date and period supplied in the VERIFIED WEATHER DATA.
8. If the user asks about current weather, now, right now, or currently, answer using CURRENT WEATHER and do not add unrelated future forecast details.
9. If the selected forecast period is now, do not mention later periods unless explicitly asked.
10. Answer in the requested language.
11. Keep the answer concise and useful.
12. Do not mention system prompts, APIs, internal instructions, implementation details, or safety classifications.
13. Do NOT output JSON unless explicitly requested.
14. Return ONLY the natural-language answer.
15. If the question is unrelated to weather, explain politely that WeatherGPT is focused on weather assistance.
""".strip()

    user_prompt = f"""
USER QUESTION:
{user_message}

DETECTED INTENT:
{intent}

REQUESTED LANGUAGE:
{language}

VERIFIED WEATHER DATA:
{_format_weather_data(weather_data)}

Now answer the USER QUESTION.
Use ONLY the verified weather data.
Return ONLY the answer that should be shown to the user.
""".strip()

    answer = await _call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    if answer.lower() == "user safety: safe" or not answer.strip():
        raise AIResponseError("AI returned an invalid weather response.")

    return AIAnswer(
        answer=answer.strip(),
        language=language,
        confidence=_calculate_answer_confidence(weather_data),
        intent=intent,
    )

# ============================================================
# CHAT HELPERS
# ============================================================

def _get_attribute(obj: object, name: str, default=None):
    return getattr(obj, name, default)


def _build_forecast_summary(forecast_data: dict) -> ForecastSummary:
    items = forecast_data.get("items", [])
    if not items:
        raise WeatherDataError("Selected forecast contains no data.")

    temperatures = [
        item["temperature"]
        for item in items
        if item.get("temperature") is not None
    ]
    rain_probabilities = [
        item["precipitation_probability"]
        for item in items
        if item.get("precipitation_probability") is not None
    ]
    precipitation_values = [
        item["precipitation"]
        for item in items
        if item.get("precipitation") is not None
    ]

    return ForecastSummary(
        date=forecast_data["target_date"],
        period=forecast_data["period"],
        timezone=forecast_data["timezone"],
        min_temperature=min(temperatures) if temperatures else None,
        max_temperature=max(temperatures) if temperatures else None,
        max_rain_probability=max(rain_probabilities) if rain_probabilities else None,
        total_precipitation=sum(precipitation_values) if precipitation_values else None,
    )


def _build_weather_context(weather_data: WeatherData) -> WeatherContext:
    current = weather_data.current
    if current is None:
        raise WeatherDataError(
            "Weather response does not contain current weather."
        )

    return WeatherContext(
        temperature=current.temperature,
        humidity=current.humidity,
        wind_speed=current.wind_speed,
        precipitation_probability=None,
        precipitation=current.precipitation or 0.0,
        weather_description=_weather_description(current.weather_code),
        observed_at=_parse_datetime(current.observed_at),
    )


def _build_chat_location(location: LocationResult) -> ChatLocation:
    return ChatLocation(
        city=location.name,
        country=location.country,
        latitude=location.latitude,
        longitude=location.longitude,
    )


def _build_ai_weather_data(
    weather_data: WeatherData,
    location: LocationResult,
    forecast_data: dict,
    alerts: list[AlertResult],
) -> dict[str, Any]:
    current = weather_data.current
    if current is None:
        raise WeatherDataError(
            "Weather response does not contain current weather."
        )

    current_data = {
        "temperature": current.temperature,
        "feels_like": current.feels_like,
        "humidity": current.humidity,
        "wind_speed": current.wind_speed,
        "wind_direction": current.wind_direction,
        "precipitation": current.precipitation,
        "pressure": current.pressure,
        "weather_code": current.weather_code,
        "description": _weather_description(current.weather_code),
        "observed_at": current.observed_at,
    }

    alert_data = [
        {
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "title": alert.title,
            "message": alert.message,
            "value": alert.value,
            "threshold": alert.threshold,
            "unit": alert.unit,
            "source": alert.source,
            "detected_at": alert.detected_at,
        }
        for alert in alerts
    ]

    return {
        "location": {
            "city": location.name,
            "country": location.country,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": location.timezone,
        },
        "current": current_data,
        "forecast": forecast_data,
        "alerts": alert_data,
    }

# ============================================================
# AUTH ROUTES
# ============================================================


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for WeatherGPT, an AI-powered conversational weather "
        "intelligence platform."
    ),
    version=settings.app_version,
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SYSTEM ROUTES
# ============================================================

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "WeatherGPT backend is running",
        "status": "online",
        "version": app.version,
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "service": "WeatherGPT API"}

# ============================================================
# AUTH ROUTES
# ============================================================

@app.post("/api/auth/register", response_model=TokenResponse, tags=["Auth"], status_code=201)
async def register_user(user_data: UserRegister):
    email = str(user_data.email).lower().strip()
    name = user_data.name.strip()
    language = user_data.language.strip().lower()

    if not name:
        raise HTTPException(400, "Name cannot be empty.")

    if get_user_by_email(email) is not None:
        raise HTTPException(409, "A user with this email already exists.")

    data = load_database()
    now = utc_now()
    user = {
        "id": next_id(data, "users"),
        "name": name,
        "email": email,
        "password_hash": hash_password(user_data.password),
        "language": language,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    data["users"].append(user)

    # Create default preferences immediately, matching the intended
    # one-to-one user preference relationship.
    data["user_preferences"].append(
        {
            "id": next_id(data, "user_preferences"),
            "user_id": user["id"],
            "timezone": settings.default_timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "notifications_enabled": True,
            "severe_alerts_enabled": True,
        }
    )
    save_database(data)

    token = create_access_token(user["id"])
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        user=UserResponse(**user),
    )


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login_user(user_data: UserLogin):
    user = get_user_by_email(str(user_data.email))
    if user is None or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect email or password.")

    if not user.get("is_active", True):
        raise HTTPException(403, "User account is inactive.")

    token = create_access_token(user["id"])
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        user=UserResponse(**user),
    )


@app.get("/api/auth/me", response_model=UserResponse, tags=["Auth"])
async def auth_me(current_user: dict[str, Any] = Depends(get_current_user)):
    return UserResponse(**current_user)

# ============================================================
# USER ROUTES
# ============================================================

@app.get("/api/users/me", response_model=UserResponse, tags=["Users"])
async def get_my_profile(current_user: dict[str, Any] = Depends(get_current_user)):
    return UserResponse(**current_user)


@app.put("/api/users/me", response_model=UserResponse, tags=["Users"])
async def update_my_profile(
    user_data: UserUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if user_data.name is None and user_data.language is None:
        raise HTTPException(
            status_code=400,
            detail="No profile fields were provided for update.",
        )

    data = load_database()
    user = next(u for u in data["users"] if u["id"] == current_user["id"])

    if user_data.name is not None:
        name = user_data.name.strip()
        if not name:
            raise HTTPException(400, "Name cannot be empty.")
        user["name"] = name

    if user_data.language is not None:
        user["language"] = user_data.language.strip()

    user["updated_at"] = utc_now()
    save_database(data)
    return UserResponse(**user)


@app.put("/api/users/me/password", tags=["Users"])
async def change_password(
    password_data: PasswordUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if not verify_password(
        password_data.current_password,
        current_user["password_hash"],
    ):
        raise HTTPException(400, "Current password is incorrect.")

    if verify_password(password_data.new_password, current_user["password_hash"]):
        raise HTTPException(
            400,
            "New password must be different from the current password.",
        )

    data = load_database()
    user = next(u for u in data["users"] if u["id"] == current_user["id"])
    user["password_hash"] = hash_password(password_data.new_password)
    user["updated_at"] = utc_now()
    save_database(data)

    return {"message": "Password changed successfully."}

# ============================================================
# PREFERENCES ROUTES
# ============================================================


def validate_timezone(timezone_name: str) -> str:
    timezone_name = timezone_name.strip()
    if not timezone_name:
        raise HTTPException(400, "Timezone cannot be empty.")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        raise HTTPException(400, f"Invalid timezone: '{timezone_name}'.")
    return timezone_name


def get_or_create_preferences(user_id: int) -> dict[str, Any]:
    data = load_database()
    preferences = next(
        (p for p in data["user_preferences"] if p["user_id"] == user_id),
        None,
    )
    if preferences is not None:
        return preferences

    preferences = {
        "id": next_id(data, "user_preferences"),
        "user_id": user_id,
        "timezone": settings.default_timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "notifications_enabled": True,
        "severe_alerts_enabled": True,
    }
    data["user_preferences"].append(preferences)
    save_database(data)
    return preferences


@app.get(
    "/api/users/me/preferences",
    response_model=UserPreferences,
    tags=["User Preferences"],
)
async def get_my_preferences(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    preferences = get_or_create_preferences(current_user["id"])
    return UserPreferences(
        language=current_user["language"],
        timezone=preferences["timezone"],
        temperature_unit=preferences["temperature_unit"],
        wind_speed_unit=preferences["wind_speed_unit"],
        notifications_enabled=preferences["notifications_enabled"],
        severe_alerts_enabled=preferences["severe_alerts_enabled"],
    )


@app.put(
    "/api/users/me/preferences",
    response_model=UserPreferences,
    tags=["User Preferences"],
)
async def update_my_preferences(
    preference_data: UserPreferences,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    timezone_name = validate_timezone(preference_data.timezone)
    data = load_database()

    user = next(u for u in data["users"] if u["id"] == current_user["id"])
    preferences = next(
        (p for p in data["user_preferences"] if p["user_id"] == current_user["id"]),
        None,
    )

    if preferences is None:
        preferences = {
            "id": next_id(data, "user_preferences"),
            "user_id": current_user["id"],
        }
        data["user_preferences"].append(preferences)

    user["language"] = preference_data.language.strip()
    user["updated_at"] = utc_now()
    preferences.update(
        {
            "timezone": timezone_name,
            "temperature_unit": preference_data.temperature_unit,
            "wind_speed_unit": preference_data.wind_speed_unit,
            "notifications_enabled": preference_data.notifications_enabled,
            "severe_alerts_enabled": preference_data.severe_alerts_enabled,
        }
    )
    save_database(data)

    return UserPreferences(
        language=user["language"],
        timezone=preferences["timezone"],
        temperature_unit=preferences["temperature_unit"],
        wind_speed_unit=preferences["wind_speed_unit"],
        notifications_enabled=preferences["notifications_enabled"],
        severe_alerts_enabled=preferences["severe_alerts_enabled"],
    )

# ============================================================
# LOCATION ROUTES
# ============================================================

async def geocode_location(city: str, country_code: str | None = None) -> LocationResult:
    city = city.strip()
    if not city:
        raise HTTPException(400, "City name cannot be empty.")
    try:
        return await geocode_city(city=city, country_code=country_code)
    except LocationNotFoundError:
        raise HTTPException(404, f"Could not find the location '{city}'.")
    except LocationAPIError:
        raise HTTPException(503, "Location service is temporarily unavailable.")
    except LocationServiceError:
        raise HTTPException(502, "Failed to retrieve location information.")


def get_user_location(user_id: int, location_id: int) -> dict[str, Any]:
    data = load_database()
    location = next(
        (
            item
            for item in data["locations"]
            if item["id"] == location_id and item["user_id"] == user_id
        ),
        None,
    )
    if location is None:
        raise HTTPException(404, "Location not found.")
    return location


@app.post(
    "/api/locations",
    response_model=LocationResponse,
    status_code=201,
    tags=["Locations"],
)
async def create_location(
    location_data: LocationCreate,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    result = await geocode_location(
        location_data.city,
        location_data.country_code,
    )
    data = load_database()

    existing = next(
        (
            item
            for item in data["locations"]
            if item["user_id"] == current_user["id"]
            and item["latitude"] == result.latitude
            and item["longitude"] == result.longitude
        ),
        None,
    )
    if existing is not None:
        raise HTTPException(409, "This location is already saved.")

    location = {
        "id": next_id(data, "locations"),
        "user_id": current_user["id"],
        "city": result.name,
        "latitude": result.latitude,
        "longitude": result.longitude,
        "country": result.country,
        "timezone": result.timezone,
        "created_at": utc_now(),
    }
    data["locations"].append(location)
    save_database(data)
    return LocationResponse(**location)


@app.get(
    "/api/locations",
    response_model=list[LocationResponse],
    tags=["Locations"],
)
async def get_my_locations(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    data = load_database()
    locations = [
        item for item in data["locations"]
        if item["user_id"] == current_user["id"]
    ]
    locations.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [LocationResponse(**item) for item in locations]


@app.get(
    "/api/locations/{location_id}",
    response_model=LocationResponse,
    tags=["Locations"],
)
async def get_location(
    location_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    return LocationResponse(**get_user_location(current_user["id"], location_id))


@app.put(
    "/api/locations/{location_id}",
    response_model=LocationResponse,
    tags=["Locations"],
)
async def update_location(
    location_id: int,
    location_data: LocationUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    result = await geocode_location(
        location_data.city,
        location_data.country_code,
    )
    data = load_database()
    location = next(
        (
            item for item in data["locations"]
            if item["id"] == location_id
            and item["user_id"] == current_user["id"]
        ),
        None,
    )
    if location is None:
        raise HTTPException(404, "Location not found.")

    existing = next(
        (
            item for item in data["locations"]
            if item["user_id"] == current_user["id"]
            and item["latitude"] == result.latitude
            and item["longitude"] == result.longitude
            and item["id"] != location_id
        ),
        None,
    )
    if existing is not None:
        raise HTTPException(409, "This location is already saved.")

    location.update(
        {
            "city": result.name,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "country": result.country,
            "timezone": result.timezone,
        }
    )
    save_database(data)
    return LocationResponse(**location)


@app.delete(
    "/api/locations/{location_id}",
    status_code=204,
    tags=["Locations"],
)
async def delete_location(
    location_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    data = load_database()
    location = next(
        (
            item for item in data["locations"]
            if item["id"] == location_id
            and item["user_id"] == current_user["id"]
        ),
        None,
    )
    if location is None:
        raise HTTPException(404, "Location not found.")

    data["locations"].remove(location)
    save_database(data)
    return None

# ============================================================
# WEATHER ROUTE HELPERS
# ============================================================

def _current_response(location: LocationResult, weather: WeatherData) -> CurrentWeatherResponse:
    current = CurrentWeather(
        temperature=weather.current.temperature,
        feels_like=weather.current.feels_like,
        humidity=weather.current.humidity or 0.0,
        wind_speed=weather.current.wind_speed or 0.0,
        wind_direction=weather.current.wind_direction,
        precipitation=weather.current.precipitation or 0.0,
        pressure=weather.current.pressure,
        weather_code=weather.current.weather_code,
        description=_weather_description(weather.current.weather_code),
        observed_at=_parse_datetime(weather.current.observed_at),
    )
    return CurrentWeatherResponse(
        location=LocationInfo(
            city=location.name,
            country=location.country,
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=location.timezone,
        ),
        current=current,
        source=weather.source,
        fetched_at=_parse_datetime(weather.fetched_at),
    )


def _forecast_response(location: LocationResult, weather: WeatherData) -> ForecastResponse:
    forecasts = [
        ForecastItem(
            forecast_time=_parse_datetime(item.time),
            temperature=item.temperature if item.temperature is not None else 0.0,
            feels_like=None,
            humidity=item.humidity,
            precipitation_probability=(
                item.precipitation_probability
                if item.precipitation_probability is not None
                else 0.0
            ),
            precipitation=item.precipitation if item.precipitation is not None else 0.0,
            wind_speed=item.wind_speed if item.wind_speed is not None else 0.0,
            weather_code=item.weather_code,
            description=_weather_description(item.weather_code),
        )
        for item in weather.hourly
    ]
    return ForecastResponse(
        location=LocationInfo(
            city=location.name,
            country=location.country,
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=location.timezone,
        ),
        forecasts=forecasts,
        source=weather.source,
        model="Open-Meteo",
        fetched_at=_parse_datetime(weather.fetched_at),
    )

# ============================================================
# WEATHER ROUTES
# ============================================================

@app.get(
    "/api/weather",
    response_model=WeatherDashboardResponse,
    tags=["Weather"],
)
async def get_city_weather(
    city: str = Query(..., min_length=1, max_length=100),
    country_code: str | None = Query(default=None, min_length=2, max_length=2),
    forecast_days: int = Query(default=7, ge=1, le=7),
):
    city = city.strip()
    if not city:
        raise HTTPException(400, "City name cannot be empty.")

    try:
        location = await geocode_city(city=city, country_code=country_code)
        weather = await get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            forecast_days=forecast_days,
        )
    except LocationNotFoundError:
        raise HTTPException(404, f"Location '{city}' was not found.")
    except LocationAPIError:
        raise HTTPException(503, "Location service is temporarily unavailable.")
    except WeatherAPIError:
        raise HTTPException(503, "Weather service is temporarily unavailable.")
    except WeatherDataError:
        raise HTTPException(502, "Weather provider returned invalid or incomplete data.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    current_response = _current_response(location, weather)
    current = current_response.current

    daily_forecast = []
    for item in weather.daily:
        daily_forecast.append(
            ForecastItem(
                forecast_time=_parse_datetime(item.date),
                temperature=item.temperature_max if item.temperature_max is not None else 0.0,
                feels_like=None,
                humidity=None,
                precipitation_probability=(
                    item.precipitation_probability_max
                    if item.precipitation_probability_max is not None
                    else 0.0
                ),
                precipitation=item.precipitation_sum if item.precipitation_sum is not None else 0.0,
                wind_speed=item.wind_speed_max if item.wind_speed_max is not None else 0.0,
                weather_code=item.weather_code,
                description=_weather_description(item.weather_code),
            )
        )

    return WeatherDashboardResponse(
        location=current_response.location,
        current=current,
        forecast=daily_forecast,
        alerts=[],
        fetched_at=current_response.fetched_at,
    )


@app.get(
    "/api/weather/current",
    response_model=CurrentWeatherResponse,
    tags=["Weather"],
)
async def get_current_city_weather(
    city: str = Query(..., min_length=1, max_length=100),
    country_code: str | None = Query(default=None, min_length=2, max_length=2),
):
    try:
        location = await geocode_city(city=city.strip(), country_code=country_code)
        weather = await get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            forecast_days=1,
        )
    except LocationNotFoundError:
        raise HTTPException(404, f"Location '{city}' was not found.")
    except (LocationAPIError, WeatherAPIError):
        raise HTTPException(503, "Unable to retrieve weather information right now.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return _current_response(location, weather)


@app.get(
    "/api/weather/forecast",
    response_model=ForecastResponse,
    tags=["Weather"],
)
async def get_city_forecast(
    city: str = Query(..., min_length=1, max_length=100),
    country_code: str | None = Query(default=None, min_length=2, max_length=2),
    forecast_days: int = Query(default=7, ge=1, le=7),
):
    try:
        location = await geocode_city(city=city.strip(), country_code=country_code)
        weather = await get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            forecast_days=forecast_days,
        )
    except LocationNotFoundError:
        raise HTTPException(404, f"Location '{city}' was not found.")
    except LocationAPIError:
        raise HTTPException(503, "Location service is unavailable.")
    except WeatherAPIError:
        raise HTTPException(503, "Weather service is unavailable.")
    except WeatherDataError:
        raise HTTPException(502, "Weather provider returned invalid or incomplete data.")

    return _forecast_response(location, weather)

# ============================================================
# ALERT ROUTE
# ============================================================

@app.get("/api/alerts", tags=["Alerts"])
async def get_alerts(
    city: str = Query(..., min_length=1, max_length=100),
):
    city = city.strip()
    if not city:
        raise HTTPException(400, "City name cannot be empty.")

    try:
        location = await geocode_city(city)
        weather_data = await get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
        )
        alerts = evaluate_weather_alerts(weather_data=weather_data)
    except LocationNotFoundError:
        raise HTTPException(404, f"Could not find location: {city}")
    except LocationAPIError:
        raise HTTPException(503, "Location service is temporarily unavailable.")
    except WeatherAPIError:
        raise HTTPException(503, "Weather service is temporarily unavailable.")
    except WeatherServiceError:
        raise HTTPException(503, "Unable to retrieve weather data.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if not alerts:
        return {
            "alert": False,
            "severity": None,
            "type": None,
            "message": "No active weather alerts.",
            "city": location.name,
            "country": location.country,
            "checked_at": datetime.now(timezone.utc),
        }

    severity_priority = {
        "low": 1,
        "moderate": 2,
        "high": 3,
        "severe": 4,
        "extreme": 5,
    }
    highest_alert = max(
        alerts,
        key=lambda alert: severity_priority.get(alert.severity.lower(), 0),
    )

    return {
        "alert": True,
        "severity": highest_alert.severity,
        "type": highest_alert.alert_type,
        "message": highest_alert.message,
        "city": location.name,
        "country": location.country,
        "checked_at": datetime.now(timezone.utc),
    }

# ============================================================
# CHAT ROUTE
# ============================================================

@app.post(
    "/api/chat",
    response_model=DetailedChatResponse,
    tags=["Chat"],
)
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(400, "Message cannot be empty.")

    try:
        intent_result = await extract_weather_intent(
            message=message,
            language=request.language,
        )
    except AIConfigurationError as exc:
        raise HTTPException(503, "AI service is not configured.") from exc
    except AIAPIError as exc:
        raise HTTPException(502, "AI service is temporarily unavailable.") from exc
    except AIResponseError as exc:
        raise HTTPException(502, "AI service returned an invalid response.") from exc
    except AIServiceError as exc:
        raise HTTPException(500, "Unable to process the weather question.") from exc

    detected_location = request.location or intent_result.location
    if not detected_location:
        raise HTTPException(
            400,
            "I could not determine the location. Please mention a city, for example 'Will it rain tomorrow in Mumbai?'",
        )

    try:
        location = await geocode_city(city=detected_location)
    except LocationNotFoundError as exc:
        raise HTTPException(
            404,
            f"Could not find the location '{detected_location}'.",
        ) from exc
    except LocationAPIError as exc:
        raise HTTPException(
            502,
            "Location service is temporarily unavailable.",
        ) from exc
    except LocationServiceError as exc:
        raise HTTPException(500, "Unable to resolve the requested location.") from exc

    try:
        weather_data = await get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            forecast_days=7,
        )
        alerts = evaluate_weather_alerts(weather_data=weather_data)

        selected_forecast = select_forecast(
            weather_data=weather_data,
            time_period=intent_result.time_period,
            timezone_name=location.timezone or "Asia/Kolkata",
        )
        forecast_items = forecast_items_to_dict(selected_forecast)
        forecast_data_context = {
            "period": selected_forecast.period,
            "target_date": str(selected_forecast.target_date),
            "timezone": selected_forecast.timezone,
            "items": forecast_items,
        }

        forecast_summary = None
        if selected_forecast.period != "now":
            forecast_summary = _build_forecast_summary(forecast_data_context)

    except ForecastNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except WeatherAPIError as exc:
        raise HTTPException(502, "Weather service is temporarily unavailable.") from exc
    except WeatherDataError as exc:
        raise HTTPException(502, "Weather service returned invalid data.") from exc
    except WeatherServiceError as exc:
        raise HTTPException(500, "Unable to retrieve weather information.") from exc

    try:
        weather_context = _build_weather_context(weather_data)
        ai_weather_data = _build_ai_weather_data(
            weather_data=weather_data,
            location=location,
            forecast_data=forecast_data_context,
            alerts=alerts,
        )
    except WeatherDataError as exc:
        raise HTTPException(502, "Unable to prepare weather information.") from exc

    try:
        answer_result = await generate_weather_answer(
            user_message=message,
            weather_data=ai_weather_data,
            language=request.language,
            intent=intent_result.intent,
        )
    except AIConfigurationError as exc:
        raise HTTPException(503, "AI service is not configured.") from exc
    except AIAPIError as exc:
        raise HTTPException(502, "AI answer service is temporarily unavailable.") from exc
    except AIResponseError as exc:
        raise HTTPException(502, "AI service returned an invalid answer.") from exc
    except AIServiceError as exc:
        raise HTTPException(500, "Unable to generate a weather answer.") from exc

    alert_response = [
        WeatherAlert(
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            value=alert.value,
            threshold=alert.threshold,
            unit=alert.unit,
            source=alert.source,
            detected_at=alert.detected_at,
        )
        for alert in alerts
    ]

    return DetailedChatResponse(
        answer=answer_result.answer,
        location=_build_chat_location(location),
        confidence=answer_result.confidence,
        weather=weather_context,
        forecast=forecast_summary,
        alerts=alert_response,
        language=answer_result.language,
        generated_at=datetime.now(timezone.utc),
    )

# ============================================================
# OPTIONAL INTERNAL AI PIPELINE
# ============================================================

async def process_weather_question(
    message: str,
    language: str = "en",
) -> WeatherIntent:
    if not message.strip():
        raise ValueError("Message cannot be empty.")
    detected_language = language
    if language == "auto":
        detected_language = await detect_language(message)
    return await extract_weather_intent(
        message=message,
        language=detected_language,
    )


print("WeatherGPT backend initialized.")
