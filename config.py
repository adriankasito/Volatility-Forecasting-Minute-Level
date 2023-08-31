"""This module extracts information from your `.env` file so that
you can use your AplhaVantage API key in other parts of the application.
"""

# The os library allows you to communicate with a computer's
# operating system: https://docs.python.org/3/library/os.html
import os

# pydantic used for data validation: https://pydantic-docs.helpmanual.io/
from pydantic_settings import BaseSettings


def return_full_path(filename: str = ".env") -> str:
    """Uses os to return the correct path of the `.env` file."""
    absolute_path = os.path.abspath(__file__)
    directory_name = os.path.dirname(absolute_path)
    full_path = os.path.join(directory_name, filename)
    return full_path


class Settings(BaseSettings):
    """Uses pydantic to define settings for project."""

    alpha_api_key: str
    db_name: str
    model_directory: str

    class Config:
        env_file = return_full_path(".env")
        protected_namespaces = ("settings_",)


# Create an instance of the Settings class with your values
settings = Settings(
alpha_api_key="1LDHCZV19GB3C9QF",
db_name="miscrosoft_corporation_financial_data",
model_directory="/Users/baba/Volatility Forecasting Project")