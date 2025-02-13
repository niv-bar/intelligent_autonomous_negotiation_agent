from pathlib import Path
import json
from typing import Optional, Dict


def load_config(file_name: str, conn: Optional[str] = None, env: Optional[str] = None) -> Optional[Dict]:
    """
    Loads a JSON configuration file from the 'config' folder, automatically adding the '.json' suffix.
    If `conn` and `env` are specified, retrieves the nested configuration for the given connection and environment.

    Parameters:
    ----------
    file_name : str
        The name of the configuration file to load (without the '.json' suffix).
    conn : str, optional
        The top-level key to access specific connection settings within the JSON file.
    env : str, optional
        The nested key under `conn` to access environment-specific settings.

    Returns:
    -------
    dict or None
        The configuration data as a dictionary if loaded successfully. If `conn` and `env` are provided,
        returns the specific nested dictionary; otherwise, returns the full configuration.

    Raises:
    ------
    FileNotFoundError:
        If the configuration file does not exist in the 'config' folder.
    JSONDecodeError:
        If the JSON file cannot be decoded.
    KeyError:
        If the specified `conn` or `env` keys are not found in the JSON structure.
    """
    # Define the configuration file path
    config_dir = Path(__file__).resolve().parent.parent / "config"
    config_path = config_dir / f"{file_name}.json"

    try:
        # Load the configuration file
        with config_path.open('r') as file:
            config = json.load(file)

        # Retrieve nested configuration if `conn` and `env` are specified
        if conn and env:
            return config[conn][env]

        return config  # Return full configuration if no specific keys are given

    except FileNotFoundError:
        print(f"Error: Config file '{file_name}.json' not found in 'config' folder.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_name}.json'.")
    except KeyError as e:
        print(f"Error: Key {e} not found in '{file_name}.json' configuration.")

    return None
