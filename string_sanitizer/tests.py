ytest
from my_module import remove_special_characters

def test_remove_special_characters():
    # Test case 1: String with special characters
    string1 = "Hello, World!"
    expected_output1 = "Hello World"
    
    assert remove_special_characters(string1) == expected_output1
    
    # Test case 2: String with alphanumeric and whitespace characters only
    string2 = "Hello123 World"
    expected_output2 = "Hello123 World"
    
    assert remove_special_characters(string2) == expected_output2
    
    # Test case 3: Empty string
    string3 = ""
    expected_output3 = ""
    
    assert remove_special_characters(string3) == expected_output3
    
    # Test case 4: String with special characters and numbers
    string4 = "Hello!123 World."
    expected_output4 = "Hello123 World"
    
    assert remove_special_characters(string4) == expected_output4
    
    # Test case 5: String with special characters and whitespace
    string5 = "Hello, \tWorld!"
    expected_output5 = "Hello \tWorld"
    
    assert remove_special_characters(string5) == expected_output5


from my_module import remove_whitespaces

def test_remove_whitespaces():
    # Test case 1: string with leading and trailing whitespaces
    assert remove_whitespaces("  hello world  ") == "hello world"
    
    # Test case 2: string with only leading whitespaces
    assert remove_whitespaces("  hello world") == "hello world"
    
    # Test case 3: string with only trailing whitespaces
    assert remove_whitespaces("hello world  ") == "hello world"
    
    # Test case 4: string without any whitespaces
    assert remove_whitespaces("helloworld") == "helloworld"
    
    # Test case 5: empty string
    assert remove_whitespaces("") == ""
    
    # Test case 6: string with multiple whitespaces in between words
    assert remove_whitespaces("hello     world") == "hello     world"


from my_module import convert_to_lowercase

# Test case 1: string with all uppercase letters
def test_convert_to_lowercase_all_uppercase():
    assert convert_to_lowercase("HELLO WORLD") == "hello world"

# Test case 2: string with all lowercase letters
def test_convert_to_lowercase_all_lowercase():
    assert convert_to_lowercase("hello world") == "hello world"

# Test case 3: string with mixed case letters
def test_convert_to_lowercase_mixed_case():
    assert convert_to_lowercase("HeLlO WoRlD") == "hello world"

# Test case 4: empty string
def test_convert_to_lowercase_empty_string():
    assert convert_to_lowercase("") == ""

# Test case 5: string with special characters and numbers
def test_convert_to_lowercase_special_characters_numbers():
    assert convert_to_lowercase("@@@123ABC") == "@@@123abc"


from my_module import remove_digits

# Import the remove_digits function
from my_module import remove_digits

# Test case: removing digits from a string with digits
def test_remove_digits_with_digits():
    assert remove_digits("Hello123World") == "HelloWorld"

# Test case: removing digits from a string without digits
def test_remove_digits_without_digits():
    assert remove_digits("HelloWorld") == "HelloWorld"

# Test case: removing digits from an empty string
def test_remove_digits_empty_string():
    assert remove_digits("") == ""

# Test case: removing digits from a string with only digits
def test_remove_digits_only_digits():
    assert remove_digits("1234567890") == ""

# Test case: removing digits from a string with special characters and digits
def test_remove_digits_special_characters_and_digits():
    assert remove_digits("!@#$%^&*()12345") == "!@#$%^&*()"

# Test case: removing digits from a string with spaces and digits
def test_remove_digits_spaces_and_digits():
    assert remove_digits("Hello 123 World") == "Hello  World"


from my_module import remove_punctuation
import string

def test_remove_punctuation_empty_string():
    assert remove_punctuation('') == ''

def test_remove_punctuation_no_punctuation():
    text = 'This is a sentence with no punctuation'
    assert remove_punctuation(text) == text

def test_remove_punctuation_all_punctuation():
    text = string.punctuation
    assert remove_punctuation(text) == ''

def test_remove_punctuation_mixed_text():
    text = 'Hello, world!'
    assert remove_punctuation(text) == 'Hello world'

def test_remove_punctuation_special_characters():
    text = 'I love Python ♥️'
    assert remove_punctuation(text) == 'I love Python '


from my_module import replace_multiple_whitespaces

def test_replace_multiple_whitespaces():
    # Test case 1: String with multiple whitespaces
    string = "Hello      world"
    expected_output = "Hello world"
    assert replace_multiple_whitespaces(string) == expected_output

    # Test case 2: String with leading and trailing whitespaces
    string = "   Hello world   "
    expected_output = " Hello world "
    assert replace_multiple_whitespaces(string) == expected_output

    # Test case 3: String with no whitespaces
    string = "HelloWorld"
    expected_output = "HelloWorld"
    assert replace_multiple_whitespaces(string) == expected_output

    # Test case 4: String with only whitespaces
    string = "       "
    expected_output = " "
    assert replace_multiple_whitespaces(string) == expected_output

    # Test case 5: Empty string
    string = ""
    expected_output = ""
    assert replace_multiple_whitespaces(string) == expected_output



from my_module import remove_stopwords
import pytest

@pytest.fixture(scope='module')
def text():
    return "This is a sample sentence."

def test_remove_stopwords(text):
    expected_output = "sample sentence."
    assert remove_stopwords(text) == expected_output

def test_remove_stopwords_with_empty_text():
    assert remove_stopwords("") == ""

def test_remove_stopwords_with_only_stopwords():
    assert remove_stopwords("the and is") == ""

def test_remove_stopwords_with_punctuation():
    assert remove_stopwords("This is a sample sentence!") == "sample sentence!"


from my_module import split_string_into_words

def test_split_string_into_words():
    # Test with a simple string
    assert split_string_into_words("Hello world") == ["Hello", "world"]

    # Test with a string containing multiple spaces between words
    assert split_string_into_words("Python   programming") == ["Python", "programming"]

    # Test with an empty string
    assert split_string_into_words("") == []

    # Test with a string containing only spaces
    assert split_string_into_words("   ") == []

    # Test with a string containing special characters
    assert split_string_into_words("Hello, world!") == ["Hello,", "world!"]

    # Test with a string containing numbers
    assert split_string_into_words("The answer is 42") == ["The", "answer", "is", "42"]

    # Test with a string containing tabs and newlines
    assert split_string_into_words("Hello\n\tworld") == ["Hello", "world"]


from my_module import join_words

# Test case 1: Empty list
def test_join_words_empty():
    assert join_words([]) == ""

# Test case 2: List with one word
def test_join_words_single_word():
    assert join_words(["hello"]) == "hello"

# Test case 3: List with multiple words
def test_join_words_multiple_words():
    assert join_words(["Hello", "world"]) == "Hello world"

# Test case 4: List with special characters and numbers
def test_join_words_special_characters():
    assert join_words(["Hello", "@", "world", "2021"]) == "Hello @ world 2021"

# Test case 5: List with empty strings
def test_join_words_empty_strings():
    assert join_words(["", "", ""]) == "  "

# Test case 6: List with whitespace strings
def test_join_words_whitespace_strings():
    assert join_words([" ", "   ", "  \t"]) == "     "

# Test case 7: List with non-string values
def test_join_words_non_string_values():
    assert join_words([1, True, None]) == "1 True None"

# Test case 8: List with mixed string and non-string values
def test_join_words_mixed_values():
    assert join_words(["Hello", 123, True, None]) == "Hello 123 True None"


from my_module import replace_words

@pytest.mark.parametrize("string, replacements, expected", [
    ("Hello world", {"world": "universe"}, "Hello universe"),  # Single word replacement
    ("The cat sat on the mat", {"cat": "dog", "mat": "rug"}, "The dog sat on the rug"),  # Multiple word replacements
    ("I love python", {"python": "programming"}, "I love programming"),  # Word replacement in a sentence
    ("12345", {"1": "one", "3": "three", "5": "five"}, "onetwothreefourfive"),  # Number replacement
    ("Hello world", {}, "Hello world"),  # No replacements
])
def test_replace_words(string, replacements, expected):
    assert replace_words(string, replacements) == expected


from my_module import sanitize_string

def test_sanitize_string():
    # Test case 1: string with special characters and uppercase letters
    assert sanitize_string("Hello! This is a Test.") == "hello this is a test"

    # Test case 2: string with only special characters
    assert sanitize_string("!@#$%^&*()") == ""

    # Test case 3: string with numbers
    assert sanitize_string("12345") == "12345"

    # Test case 4: empty string
    assert sanitize_string("") == ""

    # Test case 5: string with special characters, uppercase and lowercase letters, and numbers
    assert sanitize_string("Hello! This is a Test. 12345") == "hello this is a test 12345"


from my_module import tokenize_string

def test_tokenize_string():
    assert tokenize_string("hello world", " ") == ["hello", "world"]
    assert tokenize_string("apple,banana,cherry", ",") == ["apple", "banana", "cherry"]
    assert tokenize_string("12345;67890", ";") == ["12345", "67890"]
    assert tokenize_string("this is a test string", " ") == ["this", "is", "a", "test", "string"]
    assert tokenize_string("", ",") == []
    assert tokenize_string("hello world!", "!") == ["hello world"]


from my_module import standardize_string

@pytest.mark.parametrize('input_string, format_rules, expected_output', [
    ('hello world', None, 'Hello World'),
    (' hello world  ', None, 'Hello World'),  # leading/trailing spaces should be removed
    ('hello WORLD', ['uppercase'], 'HELLO WORLD'),  # applying uppercase rule
    ('HELLO world', ['lowercase'], 'hello world'),  # applying lowercase rule
    ('hello world', ['capitalize'], 'Hello world'),  # applying capitalize rule
    ('heLlO wOrld', ['uppercase', 'capitalize'], 'Hello World'),  # applying multiple rules
])
def test_standardize_string(input_string, format_rules, expected_output):
    assert standardize_string(input_string, format_rules) == expected_output


from my_module import extract_email_addresses

import re

def test_extract_email_addresses():
    # Test case 1: Valid email address
    text = "My email address is test@example.com"
    expected_output = ["test@example.com"]
    assert extract_email_addresses(text) == expected_output

    # Test case 2: Multiple valid email addresses
    text = "Email me at test1@example.com or test2@example.com"
    expected_output = ["test1@example.com", "test2@example.com"]
    assert extract_email_addresses(text) == expected_output

    # Test case 3: No email address in the text
    text = "This is just a plain text"
    expected_output = []
    assert extract_email_addresses(text) == expected_output

    # Test case 4: Invalid email address
    text = "My email address is not a valid email"
    expected_output = []
    assert extract_email_addresses(text) == expected_output

    # Test case 5: Email address with special characters
    text = "Email me at test.email+user@gmail.com"
    expected_output = ["test.email+user@gmail.com"]
    assert extract_email_addresses(text) == expected_output


from my_module import extract_urls

import re

def test_extract_urls_empty_string():
    string = ""
    assert extract_urls(string) == []

def test_extract_urls_no_urls():
    string = "This is a sample text without any URLs."
    assert extract_urls(string) == []

def test_extract_urls_single_url():
    string = "Check out this website: https://www.example.com"
    assert extract_urls(string) == ["https://www.example.com"]

def test_extract_urls_multiple_urls():
    string = "Visit these websites: https://www.example1.com and https://www.example2.com"
    assert extract_urls(string) == ["https://www.example1.com", "https://www.example2.com"]

def test_extract_urls_duplicate_urls():
    string = "Duplicate URLs: https://www.example.com and https://www.example.com"
    assert extract_urls(string) == ["https://www.example.com", "https://www.example.com"]

def test_extract_urls_mixed_text():
    string = "Some text with a URL: https://www.example.com and some more text."
    assert extract_urls(string) == ["https://www.example.com"]


from my_module import lemmatize_string

import pytest

@pytest.mark.parametrize('input_string, expected_output', [
    ('running', 'running'),  # Test case for lemmatizing a string with single word
    ('I am running', 'I am running'),  # Test case for lemmatizing a string with multiple words
    ('', ''),  # Test case for empty input string
    ("The cat's meow!", "The cat's meow!")  # Test case for input string with special characters
])
def test_lemmatize_string(input_string, expected_output):
    assert lemmatize_string(input_string) == expected_output


from my_module import count_word_frequency

import pytest

def test_count_word_frequency():
    # Test case 1: Empty string should return an empty dictionary
    assert count_word_frequency("") == {}

    # Test case 2: String with a single word should return a dictionary with that word and frequency 1
    assert count_word_frequency("hello") == {"hello": 1}

    # Test case 3: String with multiple words, including duplicate words, should return a dictionary with word frequencies
    assert count_word_frequency("hello world hello") == {"hello": 2, "world": 1}

    # Test case 4: String with punctuation marks should be treated as separate words
    assert count_word_frequency("Hello, world!") == {"Hello,": 1, "world!": 1}

    # Test case 5: String with leading/trailing spaces should ignore those spaces and count the words correctly
    assert count_word_frequency("   hello   world   ") == {"hello": 1, "world": 1}

    # Test case 6: String with all uppercase letters should be treated as lowercase letters
    assert count_word_frequency("HELLO WORLD hello") == {"hello": 2, "world": 1}


def test_concatenate_strings():
    # Test case 1: Test concatenation of two strings without specifying a separator
    result = concatenate_strings("Hello", "world")
    assert result == "Hello world"

    # Test case 2: Test concatenation of three strings with a specified separator
    result = concatenate_strings("Hello", "world", sep=", ")
    assert result == "Hello, world"

    # Test case 3: Test concatenation of no strings
    result = concatenate_strings()
    assert result == ""

    # Test case 4: Test concatenation of multiple strings with a different separator
    result = concatenate_strings("Hello", "world", "!", sep="-")
    assert result == "Hello-world-!"

    # Test case 5: Test concatenation of multiple empty strings with a default separator
    result = concatenate_strings("", "", "")
    assert result == " "

    # Test case 6: Test concatenation of multiple empty strings with a custom separator
    result = concatenate_strings("", "", "", sep="-")
    assert result == "--