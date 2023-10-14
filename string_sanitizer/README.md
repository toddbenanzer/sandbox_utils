# String Processing Package Documentation

This package provides a collection of functions for processing and manipulating strings in Python. It includes various utility functions that can be used to sanitize, tokenize, standardize, and extract information from strings.

## Table of Contents
- [Installation](#installation)
- [Overview](#overview)
- [Usage](#usage)
- [Examples](#examples)

## Installation

To install the package, you can use pip:

```bash
pip install string_processing_package
```

## Overview

The string processing package is designed to simplify common tasks related to string manipulation and analysis. It offers a comprehensive set of functions that can be used to clean, transform, and extract information from text data.

The main features of the package include:
- Removing special characters, digits, punctuation, and whitespaces from a string.
- Converting a string to lowercase.
- Removing stopwords (common words that do not carry much meaning).
- Splitting a string into individual words.
- Joining a list of words into a single string.
- Replacing specific words or patterns in a string.
- Sanitizing a string by removing non-alphanumeric characters.
- Tokenizing a string based on user-defined delimiters.
- Standardizing the format of a string (e.g., uppercase, lowercase, capitalized).
- Extracting email addresses and URLs from a string.
- Lemmatizing words in a string (reducing them to their base or dictionary form).
- Counting word frequency in a string.
- Concatenating multiple strings with an optional separator.

## Usage

To use the package, you need to import it into your Python script or interactive session:

```python
import string_processing_package as spp
```

Once imported, you can call any of the available functions using the `spp` namespace followed by the function name. For example:

```python
cleaned_string = spp.remove_special_characters("Hello, world!")
```

Some functions may require additional dependencies, such as the Natural Language Toolkit (NLTK) library. Please make sure to install any required dependencies before using those functions.

## Examples

Here are some examples of how to use the string processing package:

1. Cleaning and sanitizing a string:
```python
import string_processing_package as spp

dirty_string = "   Hello! This is a sample string with special characters and whitespaces.   "
clean_string = spp.remove_special_characters(dirty_string)
sanitized_string = spp.sanitize_string(clean_string)

print(sanitized_string)  # Output: hello this is a sample string with special characters and whitespaces
```

2. Tokenizing a string:
```python
import string_processing_package as spp

text = "This is a sample sentence."
tokens = spp.tokenize_string(text, delimiters=" ")

print(tokens)  # Output: ['This', 'is', 'a', 'sample', 'sentence.']
```

3. Replacing words in a string:
```python
import string_processing_package as spp

text = "I love cats and dogs."
replacements = {
    "cats": "dogs",
    "dogs": "cats"
}
replaced_text = spp.replace_words(text, replacements)

print(replaced_text)  # Output: I love dogs and cats.
```

4. Lemmatizing words in a string:
```python
import string_processing_package as spp

text = "I am running in the park."
lemmatized_text = spp.lemmatize_string(text)

print(lemmatized_text)  # Output: I am running in the park.
```

These are just a few examples of what you can do with the string processing package. Please refer to the function documentation for more details on each function's parameters and return values.