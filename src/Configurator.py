import yaml
from typing import Union, Optional


class Configurator:
    """
    A class to manage configuration settings from a YAML file.

    This class reads and retrieves configuration parameters, with optional
    string formatting using dynamic values such as `source`. It provides
    a flexible way to load configurations and retrieve parameter values for
    use in applications, allowing for custom formatting if required.

    Attributes:
        source (str): A string that can be used to format specific entries in
                      the configuration file.
        _yaml_file (dict): Internal storage for the parsed YAML configuration file.

    Methods:
        get(key: str, format_key: bool = True, **kwargs) -> Optional[Union[str, int]]:
            Retrieves a configuration value by key, with optional formatting.
    """

    def __init__(self, source: str, yaml_file_path: str):
        """
        Initializes the Configurator with a source and reads the YAML config file.

        Args:
            source (str): Source to format certain entries in the config.
            yaml_file_path (str): Path to the YAML configuration file.
        """
        self.source = source
        self._yaml_file = self._read_yaml(yaml_file_path)

    def _read_yaml(self, yaml_file_path: str) -> dict:
        """
        Reads and loads the YAML configuration file.

        Args:
            yaml_file_path (str): Path of the YAML file.

        Returns:
            dict: Parsed YAML file as a dictionary.
        """
        with open(yaml_file_path, "r") as f:
            return yaml.safe_load(f)

    def get(
        self, key: str, format_key: bool = True, **kwargs
    ) -> Optional[Union[str, int]]:
        """
        Fetches a parameter from the YAML config file.

        Args:
            key (str): Parameter to retrieve.
            format_key (bool, optional): Whether to format the entry. Defaults to True.
            **kwargs: Additional formatting parameters.

        Returns:
            Optional[Union[str, int]]: Value of the parameter, or None if key is not found.
        """
        obj = self._yaml_file.get(key)

        if obj is None:
            print(f"Warning: Key '{key}' not found in configuration.")
            return None

        if isinstance(obj, str) and format_key:
            obj = obj.format(source=self.source, **kwargs)

        return obj
