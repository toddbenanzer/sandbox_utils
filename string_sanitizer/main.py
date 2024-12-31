from typing import Dict
from typing import List
import json
import logging
import logging.handlers
import os
import re
import unicodedata
import yaml

# string_manipulation/string_sanitizer.py


class StringSanitizer:
    """
    A class for sanitizing strings by removing unwanted characters, normalizing Unicode,
    cleaning whitespace, and stripping specified characters.
    """

    def __init__(self, options: Dict[str, any]) -> None:
        """
        Initializes the StringSanitizer with specific options for sanitization.

        Args:
            options (Dict[str, any]): Configuration options for sanitization.
        """
        self.options = options

    def remove_unwanted_characters(self, text: str) -> str:
        """
        Removes characters considered unwanted from the input string.

        Args:
            text (str): The input string from which unwanted characters are to be removed.

        Returns:
            str: A new string with unwanted characters removed based on the options.
        """
        unwanted_chars = self.options.get("unwanted_chars", "")
        return ''.join(ch for ch in text if ch not in unwanted_chars)

    def normalize_unicode(self, text: str, form: str = 'NFC') -> str:
        """
        Normalizes Unicode characters in the string to the specified form.

        Args:
            text (str): The input string to be normalized.
            form (str): The normalization form (e.g., 'NFC', 'NFD'). Defaults to 'NFC'.

        Returns:
            str: A string normalized to the specified Unicode form.
        """
        return unicodedata.normalize(form, text)

    def clean_whitespace(self, text: str) -> str:
        """
        Trims and removes unnecessary whitespace from the string.

        Args:
            text (str): The input string to clean whitespace.

        Returns:
            str: A string with excess whitespace removed.
        """
        return ' '.join(text.split())

    def strip_characters(self, text: str, characters: str) -> str:
        """
        Strips specified characters from both ends of the string.

        Args:
            text (str): The input string.
            characters (str): A string of characters to be removed from the ends of `text`.

        Returns:
            str: The string with specified leading and trailing characters removed.
        """
        return text.strip(characters)


# string_manipulation/string_tokenizer.py


class StringTokenizer:
    """
    A class for tokenizing strings using specified delimiters and regular expression patterns.
    """

    def __init__(self, delimiters: List[str], token_patterns: List[str]) -> None:
        """
        Initializes the StringTokenizer with specified delimiters and token patterns.

        Args:
            delimiters (List[str]): A list of characters or substrings that define where tokens begin and end within strings.
            token_patterns (List[str]): A list of regular expressions representing token patterns.
        """
        self.delimiters = delimiters
        self.token_patterns = token_patterns

    def split_into_tokens(self, text: str) -> List[str]:
        """
        Splits the input string into a list of tokens based on the defined delimiters and token patterns.

        Args:
            text (str): The input string to be tokenized.

        Returns:
            List[str]: A list of tokens extracted from the input string.
        """
        delimiter_regex = '|'.join(map(re.escape, self.delimiters))
        raw_tokens = re.split(delimiter_regex, text)

        tokens = []
        for raw_token in raw_tokens:
            for pattern in self.token_patterns:
                matched_tokens = re.findall(pattern, raw_token)
                tokens.extend(matched_tokens)
        
        return tokens

    def customize_token_patterns(self, new_patterns: List[str]) -> None:
        """
        Allows users to update or add new token patterns for tokenization.

        Args:
            new_patterns (List[str]): A list of new or additional regular expression patterns for token identification.
        """
        self.token_patterns.extend(new_patterns)


# string_manipulation/string_standardizer.py


class StringStandardizer:
    """
    A class for standardizing strings by converting case, mapping to standardized
    forms, and enforcing specific formats.
    """

    def __init__(self, standard_maps: Dict[str, Dict[str, str]]) -> None:
        """
        Initializes the StringStandardizer with specified standardization mappings.

        Args:
            standard_maps (Dict[str, Dict[str, str]]): Dictionary containing mappings or
            rules for standardizing text.
        """
        self.standard_maps = standard_maps

    def convert_case(self, text: str, case_type: str) -> str:
        """
        Converts the input string to a specified case based on `case_type`.

        Args:
            text (str): The input string to change the case.
            case_type (str): The desired case type ('lower', 'upper', 'title').

        Returns:
            str: The string converted to the specified case type.
        """
        if case_type == 'lower':
            return text.lower()
        elif case_type == 'upper':
            return text.upper()
        elif case_type == 'title':
            return text.title()
        else:
            raise ValueError(f"Unsupported case type: {case_type}")

    def map_to_standard(self, text: str, map_type: str) -> str:
        """
        Maps input strings to their standardized versions based on specified mapping rules.

        Args:
            text (str): The input string to be standardized.
            map_type (str): The key to identify which mapping to use from `standard_maps`.

        Returns:
            str: The standardized string according to the mapping rules.
        """
        mapping = self.standard_maps.get(map_type, {})
        return mapping.get(text, text)

    def enforce_format(self, text: str, format_type: str) -> str:
        """
        Ensures that strings adhere to a specified format, modifying them as necessary.

        Args:
            text (str): The input string to be formatted.
            format_type (str): The specified format rule or guideline that needs to be enforced.

        Returns:
            str: The formatted string that follows the specified rules or guidelines.
        """
        # Assuming a predefined set of format rules; users can expand this method
        if format_type == 'email':
            return text.lower().strip()
        elif format_type == 'phone':
            return ''.join(filter(str.isdigit, text))
        else:
            raise ValueError(f"Unsupported format type: {format_type}")


# string_manipulation/marketing_pattern_utility.py


class MarketingPatternUtility:
    """
    A utility class for matching and replacing patterns commonly found in marketing data.
    """

    def __init__(self) -> None:
        """
        Initializes the MarketingPatternUtility instance.
        """
        # Define regex patterns for different pattern types
        self.patterns = {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "url": r"https?://(?:www\.)?\S+(?:/\S+)*",
            "phone": r"\+?1?\d{9,15}",
        }

    def match_pattern(self, text: str, pattern_type: str) -> List[str]:
        """
        Searches the input string for matches to the specified pattern type.

        Args:
            text (str): The input string in which to search for the pattern.
            pattern_type (str): A key indicating the type of pattern to use for matching.

        Returns:
            List[str]: A list of all matches found in the string based on the specified pattern type.
        """
        pattern = self.patterns.get(pattern_type)
        if not pattern:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
        return re.findall(pattern, text)

    def replace_pattern(self, text: str, pattern_type: str, replacement: str) -> str:
        """
        Replaces all occurrences of the specified pattern type in the input string with a replacement string.

        Args:
            text (str): The input string containing the pattern to be replaced.
            pattern_type (str): A key indicating the type of pattern to find and replace.
            replacement (str): The string to substitute for each occurrence of the pattern.

        Returns:
            str: The modified string with pattern replacements performed.
        """
        pattern = self.patterns.get(pattern_type)
        if not pattern:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
        return re.sub(pattern, replacement, text)


# string_manipulation/file_utils.py

def load_string_data(file_path: str) -> str:
    """
    Loads and reads string data from a specified file path.

    Args:
        file_path (str): The path to the file containing string data to be loaded.

    Returns:
        str: The content of the file as a single string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an error reading the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path '{file_path}' was not found.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file: {e}")


# string_manipulation/file_utils.py

def save_string_data(data: str, file_path: str) -> None:
    """
    Saves string data to a specified file path.

    Args:
        data (str): The string data to be written to the file.
        file_path (str): The path to the file where string data should be saved.

    Raises:
        IOError: If there is an error writing to the file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(data)
    except IOError as e:
        raise IOError(f"An error occurred while writing to the file: {e}")


# string_manipulation/logging_utils.py


def setup_logging(level: str) -> None:
    """
    Configures the logging system for the application to use the specified logging level.

    Args:
        level (str): Defines the logging level to be set for the application.

    Raises:
        ValueError: If an invalid logging level is provided.
    """
    # Define valid logging levels
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    if level not in levels:
        raise ValueError(f"Invalid logging level specified: {level}")

    # Set up the logging format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Configure logging to console
    logging.basicConfig(level=levels[level], format=log_format)

    # Configure logging to file
    log_file_handler = logging.handlers.RotatingFileHandler(
        "application.log", maxBytes=10*1024*1024, backupCount=5
    )
    log_file_handler.setFormatter(logging.Formatter(log_format))

    # Add file handler to the root logger
    logger = logging.getLogger()
    logger.addHandler(log_file_handler)


# string_manipulation/config_utils.py


def setup_config(config_file: str) -> dict:
    """
    Reads and parses a configuration file providing the application with necessary configuration settings.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing configuration settings parsed from the file.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If there is an error parsing the configuration file.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file '{config_file}' was not found.")

    try:
        with open(config_file, 'r') as file:
            if config_file.endswith('.json'):
                return json.load(file)
            elif config_file.endswith(('.yml', '.yaml')):
                return yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file}")
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Error parsing the configuration file: {e}")
