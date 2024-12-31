from string_manipulation.config_utils import setup_config
from string_manipulation.file_utils import load_string_data
from string_manipulation.file_utils import save_string_data
from string_manipulation.logging_utils import setup_logging
from string_manipulation.marketing_pattern_utility import MarketingPatternUtility
from string_manipulation.string_sanitizer import StringSanitizer
from string_manipulation.string_standardizer import StringStandardizer
from string_manipulation.string_tokenizer import StringTokenizer
import logging

# Example usage of StringSanitizer


# Initialize with unwanted characters
options = {
    "unwanted_chars": "!@#"
}
sanitizer = StringSanitizer(options)

# Remove unwanted characters
result1 = sanitizer.remove_unwanted_characters("Hello! World@ #2023")
print(result1)  # Output: "Hello World 2023"

# Normalize Unicode characters
result2 = sanitizer.normalize_unicode("Café", form='NFC')
print(result2)  # Output: "Café"

result3 = sanitizer.normalize_unicode("Café", form='NFD')
print(result3)  # Output: "Café"

# Clean excess whitespace
result4 = sanitizer.clean_whitespace("   Well   spaced   text  ")
print(result4)  # Output: "Well spaced text"

# Strip specific characters
result5 = sanitizer.strip_characters("##Python##", "#")
print(result5)  # Output: "Python"


# Example usage of StringTokenizer


# Initialize with delimiters and token patterns
delimiters = [",", ";", ":"]
token_patterns = [r"\b\w+\b", r"\d+"]
tokenizer = StringTokenizer(delimiters, token_patterns)

# Tokenize a string
text = "Hello, world: this is a test; 1234"
tokens = tokenizer.split_into_tokens(text)
print(tokens)  # Output: ['Hello', 'world', 'this', 'is', 'a', 'test', '1234']

# Add new token patterns
new_patterns = [r"#\w+", r"\$\d+"]
tokenizer.customize_token_patterns(new_patterns)

# Tokenize a string with new patterns
text2 = "Get #discount for $5 off your order; use code #save"
tokens2 = tokenizer.split_into_tokens(text2)
print(tokens2)  # Output: ['Get', 'discount', 'for', '5', 'off', 'your', 'order', 'use', 'code', 'save', '#discount', '$5', '#save']

# Tokenize a string without delimiters in it
text3 = "SimplyNoDelimitersHere"
tokens3 = tokenizer.split_into_tokens(text3)
print(tokens3)  # Output: ['SimplyNoDelimitersHere']


# Example usage of StringStandardizer


# Initialize with standardization mappings
standard_maps = {
    "brand_names": {"appl": "Apple", "msft": "Microsoft"},
    "units": {"kg": "kilogram", "g": "gram"}
}
standardizer = StringStandardizer(standard_maps)

# Convert case example
text1 = "hello World"
print(standardizer.convert_case(text1, 'upper'))     # Output: "HELLO WORLD"
print(standardizer.convert_case(text1, 'lower'))     # Output: "hello world"
print(standardizer.convert_case(text1, 'title'))     # Output: "Hello World"

# Map to standard example
abbreviation = "appl"
print(standardizer.map_to_standard(abbreviation, "brand_names"))  # Output: "Apple"

# Enforce format example
email = "  EXAMPLE@Domain.COM  "
print(standardizer.enforce_format(email, "email"))  # Output: "example@domain.com"

phone = "(123) 456-7890"
print(standardizer.enforce_format(phone, "phone"))  # Output: "1234567890"


# Example usage of MarketingPatternUtility


# Initialize the utility
pattern_utility = MarketingPatternUtility()

# Match patterns in text
text_with_emails = "For inquiries, email us at contact@example.com or sales@domain.org."
emails = pattern_utility.match_pattern(text_with_emails, "email")
print(emails)  # Output: ['contact@example.com', 'sales@domain.org']

# Replace patterns in text
text_with_urls = "Visit https://www.example.com and our blog at http://blog.example.com"
updated_text_with_urls = pattern_utility.replace_pattern(text_with_urls, "url", "[LINK]")
print(updated_text_with_urls)  # Output: "Visit [LINK] and our blog at [LINK]"

# Match phone numbers in text
text_with_phones = "Reach us at +1234567890 or 987654321."
phones = pattern_utility.match_pattern(text_with_phones, "phone")
print(phones)  # Output: ['+1234567890', '987654321']

# Replace phone numbers in text
updated_text_with_phones = pattern_utility.replace_pattern(text_with_phones, "phone", "[PHONE]")
print(updated_text_with_phones)  # Output: "Reach us at [PHONE] or [PHONE]"


# Example usage of load_string_data


# Example 1: Load a text file containing plain text
try:
    content = load_string_data('example.txt')
    print(content)
except FileNotFoundError as e:
    print(e)
except IOError as e:
    print(e)

# Example 2: Handle file not found error
try:
    content = load_string_data('non_existent_file.txt')
except FileNotFoundError as e:
    print(f"Error: {e}")

# Example 3: Handle IO error
try:
    content = load_string_data('/path/to/restricted_access_file.txt')
except IOError as e:
    print(f"An IO error occurred: {e}")


# Example usage of save_string_data


# Example 1: Save a simple string to a file
content = "This is a sample text to be saved to a file."
file_path = "sample_output.txt"
try:
    save_string_data(content, file_path)
    print(f"Data successfully saved to {file_path}")
except IOError as e:
    print(e)

# Example 2: Save data to a file in a specific directory
content = "This is another example of saving text."
file_path = "/path/to/directory/example_output.txt"
try:
    save_string_data(content, file_path)
    print(f"Data successfully saved to {file_path}")
except IOError as e:
    print(e)

# Example 3: Save large string data to a file
large_content = "This is a large content. " * 1000  # Large text
large_file_path = "large_output.txt"
try:
    save_string_data(large_content, large_file_path)
    print(f"Large data successfully saved to {large_file_path}")
except IOError as e:
    print(e)


# Example usage of setup_logging


# Example 1: Set up logging with INFO level
setup_logging("INFO")
logging.info("This is an informational message.")

# Example 2: Set up logging with DEBUG level
setup_logging("DEBUG")
logging.debug("This is a debug message.")

# Example 3: Attempting an invalid logging level
try:
    setup_logging("INVALID")
except ValueError as e:
    print(e)

# Example 4: Set up logging with ERROR level and log an error
setup_logging("ERROR")
logging.error("This is an error message.")


# Example usage of setup_config


# Example 1: Load configuration from a JSON file
try:
    config = setup_config('config.json')
    print("JSON Config Loaded:", config)
except (FileNotFoundError, ValueError) as e:
    print(e)

# Example 2: Load configuration from a YAML file
try:
    config = setup_config('config.yml')
    print("YAML Config Loaded:", config)
except (FileNotFoundError, ValueError) as e:
    print(e)

# Example 3: Handling a file not found situation
try:
    config = setup_config('non_existent_config.yaml')
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Example 4: Handling unsupported file format
try:
    config = setup_config('config.txt')
except ValueError as e:
    print(f"Unsupported format: {e}")
