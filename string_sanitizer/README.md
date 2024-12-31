# StringSanitizer Documentation

## Class: StringSanitizer

### Description
A class for sanitizing strings by removing unwanted characters, normalizing Unicode, cleaning whitespace, and stripping specified characters.

### Constructor

#### `__init__(self, options: Dict[str, any])`
Initializes the StringSanitizer with specific options for sanitization.

- **Parameters:**
  - `options` (Dict[str, any]): Configuration options for sanitization. This may include keys such as `unwanted_chars` that define which characters should be removed during the sanitization process.

### Methods

#### `remove_unwanted_characters(self, text: str) -> str`
Removes characters considered unwanted from the input string.

- **Parameters:**
  - `text` (str): The input string from which unwanted characters are to be removed.

- **Returns:**
  - str: A new string with unwanted characters removed based on the options specified during initialization.

---

#### `normalize_unicode(self, text: str, form: str = 'NFC') -> str`
Normalizes Unicode characters in the string to the specified form.

- **Parameters:**
  - `text` (str): The input string to be normalized.
  - `form` (str): The normalization form (e.g., 'NFC', 'NFD'). Defaults to 'NFC'.

- **Returns:**
  - str: A string normalized to the specified Unicode form.

---

#### `clean_whitespace(self, text: str) -> str`
Trims and removes unnecessary whitespace from the string.

- **Parameters:**
  - `text` (str): The input string to clean whitespace.

- **Returns:**
  - str: A string with excess whitespace removed.

---

#### `strip_characters(self, text: str, characters: str) -> str`
Strips specified characters from both ends of the string.

- **Parameters:**
  - `text` (str): The input string.
  - `characters` (str): A string of characters to be removed from the ends of `text`.

- **Returns:**
  - str: The string with specified leading and trailing characters removed.


# StringTokenizer Documentation

## Class: StringTokenizer

### Description
A class for tokenizing strings using specified delimiters and regular expression patterns.

### Constructor

#### `__init__(self, delimiters: List[str], token_patterns: List[str])`
Initializes the StringTokenizer with specified delimiters and token patterns.

- **Parameters:**
  - `delimiters` (List[str]): A list of characters or substrings that define where tokens begin and end within strings.
  - `token_patterns` (List[str]): A list of regular expressions representing token patterns.

### Methods

#### `split_into_tokens(self, text: str) -> List[str]`
Splits the input string into a list of tokens based on the defined delimiters and token patterns.

- **Parameters:**
  - `text` (str): The input string to be tokenized.

- **Returns:**
  - List[str]: A list of tokens extracted from the input string.

---

#### `customize_token_patterns(self, new_patterns: List[str]) -> None`
Allows users to update or add new token patterns for tokenization.

- **Parameters:**
  - `new_patterns` (List[str]): A list of new or additional regular expression patterns for token identification.


# StringStandardizer Documentation

## Class: StringStandardizer

### Description
A class for standardizing strings by converting case, mapping to standardized forms, and enforcing specific formats.

### Constructor

#### `__init__(self, standard_maps: Dict[str, Dict[str, str]])`
Initializes the StringStandardizer with specified standardization mappings.

- **Parameters:**
  - `standard_maps` (Dict[str, Dict[str, str]]): Dictionary containing mappings or rules for standardizing text.

### Methods

#### `convert_case(self, text: str, case_type: str) -> str`
Converts the input string to a specified case based on `case_type`.

- **Parameters:**
  - `text` (str): The input string to change the case.
  - `case_type` (str): The desired case type ('lower', 'upper', 'title').

- **Returns:**
  - str: The string converted to the specified case type.

---

#### `map_to_standard(self, text: str, map_type: str) -> str`
Maps input strings to their standardized versions based on specified mapping rules.

- **Parameters:**
  - `text` (str): The input string to be standardized.
  - `map_type` (str): The key to identify which mapping to use from `standard_maps`.

- **Returns:**
  - str: The standardized string according to the mapping rules.

---

#### `enforce_format(self, text: str, format_type: str) -> str`
Ensures that strings adhere to a specified format, modifying them as necessary.

- **Parameters:**
  - `text` (str): The input string to be formatted.
  - `format_type` (str): The specified format rule or guideline that needs to be enforced.

- **Returns:**
  - str: The formatted string that follows the specified rules or guidelines.


# MarketingPatternUtility Documentation

## Class: MarketingPatternUtility

### Description
A utility class for matching and replacing patterns commonly found in marketing data.

### Constructor

#### `__init__(self)`
Initializes the MarketingPatternUtility instance.

- **Parameters:** None

### Methods

#### `match_pattern(self, text: str, pattern_type: str) -> List[str]`
Searches the input string for matches to the specified pattern type.

- **Parameters:**
  - `text` (str): The input string in which to search for the pattern.
  - `pattern_type` (str): A key indicating the type of pattern to use for matching. Supported types include:
    - `email`: Matches email addresses.
    - `url`: Matches URLs.
    - `phone`: Matches phone numbers.

- **Returns:**
  - List[str]: A list of all matches found in the string based on the specified pattern type.

- **Raises:**
  - ValueError: If the given `pattern_type` is unsupported.

---

#### `replace_pattern(self, text: str, pattern_type: str, replacement: str) -> str`
Replaces all occurrences of the specified pattern type in the input string with a replacement string.

- **Parameters:**
  - `text` (str): The input string containing the pattern to be replaced.
  - `pattern_type` (str): A key indicating the type of pattern to find and replace. Supported types include:
    - `email`: Matches email addresses.
    - `url`: Matches URLs.
    - `phone`: Matches phone numbers.
  - `replacement` (str): The string to substitute for each occurrence of the pattern.

- **Returns:**
  - str: The modified string with pattern replacements performed.

- **Raises:**
  - ValueError: If the given `pattern_type` is unsupported.


# load_string_data Documentation

## Function: load_string_data

### Description
Loads and reads string data from a specified file path.

### Parameters

#### `file_path`
- **Type:** `str`
- **Description:** The path to the file containing string data to be loaded.

### Returns
- **Type:** `str`
- **Description:** The content of the file as a single string.

### Raises
- **FileNotFoundError:** If the specified file does not exist.
- **IOError:** If there is an error reading the file.

### Usage Example


# save_string_data Documentation

## Function: save_string_data

### Description
Saves string data to a specified file path.

### Parameters

#### `data`
- **Type:** `str`
- **Description:** The string data to be written to the file.

#### `file_path`
- **Type:** `str`
- **Description:** The path to the file where string data should be saved.

### Returns
- **Type:** `None`
- **Description:** The function does not return a value.

### Raises
- **IOError:** If there is an error writing to the file.

### Usage Example


# setup_logging Documentation

## Function: setup_logging

### Description
Configures the logging system for the application to use the specified logging level.

### Parameters

#### `level`
- **Type:** `str`
- **Description:** Defines the logging level to be set for the application. The valid logging levels are:
  - `"DEBUG"`: Detailed information, typically of interest only when diagnosing problems.
  - `"INFO"`: Confirm that things are working as expected.
  - `"WARNING"`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., ‘disk space low’).
  - `"ERROR"`: Due to a more serious problem, the software has not been able to perform some function.
  - `"CRITICAL"`: A very serious error, indicating that the program itself may be unable to continue running.

### Returns
- **Type:** `None`
- **Description:** This function does not return a value.

### Raises
- **ValueError:** If an invalid logging level is provided.

### Usage Example


# setup_config Documentation

## Function: setup_config

### Description
Reads and parses a configuration file, providing the application with necessary configuration settings.

### Parameters

#### `config_file`
- **Type:** `str`
- **Description:** The path to the configuration file. This file should be in a supported format (JSON or YAML).

### Returns
- **Type:** `dict`
- **Description:** A dictionary containing configuration settings parsed from the file.

### Raises
- **FileNotFoundError:** If the specified configuration file does not exist.
- **ValueError:** If there is an error parsing the configuration file, including unsupported file formats.

### Usage Example
