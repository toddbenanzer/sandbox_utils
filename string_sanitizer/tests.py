from string_manipulation.config_utils import setup_config
from string_manipulation.file_utils import load_string_data
from string_manipulation.file_utils import save_string_data
from string_manipulation.logging_utils import setup_logging
from string_manipulation.marketing_pattern_utility import MarketingPatternUtility
from string_manipulation.string_sanitizer import StringSanitizer
from string_manipulation.string_standardizer import StringStandardizer
from string_manipulation.string_tokenizer import StringTokenizer
import json
import logging
import os
import pytest
import yaml

# test_string_sanitizer.py


@pytest.fixture
def sanitizer():
    options = {
        "unwanted_chars": "!@#"
    }
    return StringSanitizer(options)

def test_remove_unwanted_characters(sanitizer):
    assert sanitizer.remove_unwanted_characters("Hello! World@") == "Hello World"
    assert sanitizer.remove_unwanted_characters("123#456") == "123456"
    assert sanitizer.remove_unwanted_characters("NoSpecialChar") == "NoSpecialChar"

def test_normalize_unicode(sanitizer):
    assert sanitizer.normalize_unicode("Café", form='NFD') == 'Café'
    assert sanitizer.normalize_unicode("Café", form='NFC') == 'Café'

def test_clean_whitespace(sanitizer):
    assert sanitizer.clean_whitespace("   Hello   World  ") == "Hello World"
    assert sanitizer.clean_whitespace("No  Spaces") == "No Spaces"
    assert sanitizer.clean_whitespace("   Single") == "Single"

def test_strip_characters(sanitizer):
    assert sanitizer.strip_characters("##Hello##", "#") == "Hello"
    assert sanitizer.strip_characters("!!!Hello!!!", "!") == "Hello"
    assert sanitizer.strip_characters("NoStrip", "@") == "NoStrip"


# test_string_tokenizer.py


@pytest.fixture
def tokenizer():
    delimiters = [",", ";", ":"]
    token_patterns = [r"\b\w+\b", r"\d+"]
    return StringTokenizer(delimiters, token_patterns)

def test_split_into_tokens(tokenizer):
    text = "Hello, world: this is a test; 1234"
    tokens = tokenizer.split_into_tokens(text)
    expected_tokens = ["Hello", "world", "this", "is", "a", "test", "1234"]
    assert tokens == expected_tokens

def test_split_with_additional_delimiters(tokenizer):
    text = "apple;banana:orange,grape"
    tokens = tokenizer.split_into_tokens(text)
    expected_tokens = ["apple", "banana", "orange", "grape"]
    assert tokens == expected_tokens

def test_customize_token_patterns(tokenizer):
    new_patterns = [r"#\w+", r"\$\d+"]
    tokenizer.customize_token_patterns(new_patterns)
    
    text = "Get #discount for $5 off your order; use code #save"
    tokens = tokenizer.split_into_tokens(text)
    expected_tokens = ["Get", "discount", "for", "5", "off", "your", "order", "use", "code", "save", "#discount", "$5", "#save"]
    assert tokens == expected_tokens

def test_no_delimiters(tokenizer):
    text = "SimpleNoDelimiters"
    tokens = tokenizer.split_into_tokens(text)
    expected_tokens = ["SimpleNoDelimiters"]
    assert tokens == expected_tokens


# test_string_standardizer.py


@pytest.fixture
def standardizer():
    standard_maps = {
        "brand_names": {"appl": "Apple", "msft": "Microsoft"},
        "unit_measurement": {"kg": "kilogram", "m": "meter"}
    }
    return StringStandardizer(standard_maps)

def test_convert_case(standardizer):
    assert standardizer.convert_case("hello world", "upper") == "HELLO WORLD"
    assert standardizer.convert_case("HELLO WORLD", "lower") == "hello world"
    assert standardizer.convert_case("hello world", "title") == "Hello World"
    with pytest.raises(ValueError):
        standardizer.convert_case("hello", "invalid")

def test_map_to_standard(standardizer):
    assert standardizer.map_to_standard("appl", "brand_names") == "Apple"
    assert standardizer.map_to_standard("msft", "brand_names") == "Microsoft"
    assert standardizer.map_to_standard("nflx", "brand_names") == "nflx"

def test_enforce_format(standardizer):
    assert standardizer.enforce_format(" John.Doe@MAIL.com ", "email") == "john.doe@mail.com"
    assert standardizer.enforce_format(" (123) 456-7890 ", "phone") == "1234567890"
    with pytest.raises(ValueError):
        standardizer.enforce_format("content", "unknown_format")


# test_marketing_pattern_utility.py


@pytest.fixture
def pattern_utility():
    return MarketingPatternUtility()

def test_match_pattern_email(pattern_utility):
    text = "Contact us at info@example.com or support@domain.com."
    matches = pattern_utility.match_pattern(text, "email")
    expected = ["info@example.com", "support@domain.com"]
    assert matches == expected

def test_match_pattern_url(pattern_utility):
    text = "Check out our website at https://www.example.com or http://domain.com/page."
    matches = pattern_utility.match_pattern(text, "url")
    expected = ["https://www.example.com", "http://domain.com/page"]
    assert matches == expected

def test_match_pattern_phone(pattern_utility):
    text = "Call us at +1234567890 or 0987654321 for more information."
    matches = pattern_utility.match_pattern(text, "phone")
    expected = ["+1234567890", "0987654321"]
    assert matches == expected

def test_replace_pattern_email(pattern_utility):
    text = "Please contact us at info@example.com."
    result = pattern_utility.replace_pattern(text, "email", "[EMAIL]")
    expected = "Please contact us at [EMAIL]."
    assert result == expected

def test_replace_pattern_url(pattern_utility):
    text = "Visit our site at https://www.example.com for updates."
    result = pattern_utility.replace_pattern(text, "url", "[URL]")
    expected = "Visit our site at [URL] for updates."
    assert result == expected

def test_replace_pattern_phone(pattern_utility):
    text = "For info, call +1234567890."
    result = pattern_utility.replace_pattern(text, "phone", "[PHONE]")
    expected = "For info, call [PHONE]."
    assert result == expected

def test_unsupported_pattern_type(pattern_utility):
    text = "Some random text"
    with pytest.raises(ValueError):
        pattern_utility.match_pattern(text, "unsupported")
    with pytest.raises(ValueError):
        pattern_utility.replace_pattern(text, "unsupported", "[REPLACEMENT]")


# test_file_utils.py


def test_load_string_data_success(tmp_path):
    # Create a temporary file
    temp_file = tmp_path / "test_file.txt"
    content = "This is a test file with some text."
    temp_file.write_text(content, encoding='utf-8')
    
    # Test loading the file content
    result = load_string_data(str(temp_file))
    assert result == content

def test_load_string_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_string_data("non_existent_file.txt")

def test_load_string_data_io_error(mocker):
    temp_file = "test_io_error.txt"
    mocker.patch("builtins.open", side_effect=IOError("Mocked IO error"))
    
    with pytest.raises(IOError):
        load_string_data(temp_file)

def test_load_string_data_with_encoding(tmp_path):
    temp_file = tmp_path / "test_file_utf16.txt"
    content = "This is a UTF-16 encoded test file."
    temp_file.write_text(content, encoding='utf-16')
    
    with pytest.raises(UnicodeDecodeError):
        load_string_data(str(temp_file))


# test_file_utils.py


def test_save_string_data_success(tmp_path):
    # Create a temporary file path
    temp_file = tmp_path / "test_output.txt"
    content = "This is a string to be saved to a file."

    # Test saving string data
    save_string_data(content, str(temp_file))
    
    # Read the file and check content
    with open(str(temp_file), 'r', encoding='utf-8') as file:
        result = file.read()
    assert result == content

def test_save_string_data_io_error(mocker):
    mocker.patch("builtins.open", side_effect=IOError("Mocked IO error"))
    
    with pytest.raises(IOError):
        save_string_data("Some data", "dummy_path/test_output.txt")


# test_logging_utils.py


def test_setup_logging_valid_levels(caplog):
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    for level in valid_levels:
        setup_logging(level)
        logger = logging.getLogger()
        logger.log(getattr(logging, level), f"Test log at {level} level")
        assert level in caplog.text

def test_setup_logging_invalid_level():
    with pytest.raises(ValueError, match="Invalid logging level specified: INVALID"):
        setup_logging("INVALID")

def test_logging_file_creation(tmp_path, monkeypatch):
    log_file = tmp_path / "test_application.log"
    
    # Monkeypatch the RotatingFileHandler to use the temporary path
    monkeypatch.setattr(logging.handlers, "RotatingFileHandler",
                        lambda *args, **kwargs: logging.FileHandler(log_file))
    
    setup_logging("INFO")
    logger = logging.getLogger()
    logger.info("Info level log test")
    
    assert log_file.read_text().strip() != ""


# test_config_utils.py


def test_setup_config_json(tmp_path):
    # Create a temporary JSON config file
    config_data = {"setting1": "value1", "setting2": "value2"}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data), encoding='utf-8')

    # Test parsing the JSON config file
    result = setup_config(str(config_file))
    assert result == config_data

def test_setup_config_yaml(tmp_path):
    # Create a temporary YAML config file
    config_data = {"settingA": "valueA", "settingB": "valueB"}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data), encoding='utf-8')

    # Test parsing the YAML config file
    result = setup_config(str(config_file))
    assert result == config_data

def test_setup_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        setup_config("non_existent_config.json")

def test_setup_config_invalid_json(tmp_path):
    # Create a malformed JSON config file
    config_file = tmp_path / "invalid_config.json"
    config_file.write_text("{invalid_json:}", encoding='utf-8')

    with pytest.raises(ValueError):
        setup_config(str(config_file))

def test_setup_config_unsupported_format(tmp_path):
    # Create a config file with an unsupported extension
    config_file = tmp_path / "config.txt"
    config_file.write_text("some unsupported content", encoding='utf-8')

    with pytest.raises(ValueError):
        setup_config(str(config_file))
