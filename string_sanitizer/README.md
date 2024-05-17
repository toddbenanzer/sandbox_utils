```markdown
# Python Text Manipulation Package

## Overview
This Python package provides a collection of functions for text manipulation. It includes functionalities such as removing special characters, converting case, extracting email addresses and URLs, calculating string similarity scores, and more.

## Usage
To utilize this package, you can simply import the functions you need into your Python script. Here's an example:

```python
from text_manipulation import remove_special_characters

string = "Hello! This is a test string."
clean_string = remove_special_characters(string)
print(clean_string)  # Output: "Hello This is a test string"
```

## Examples
Here are some examples of the main functionalities provided by this package:

### Removing special characters
```python
from text_manipulation import remove_special_characters

string = "Hello! This is a test string."
clean_string = remove_special_characters(string)
print(clean_string)  # Output: "Hello This is a test string"
```

### Converting to lowercase
```python
from text_manipulation import convert_to_lowercase

string = "Hello! This is a test string."
lowercase_string = convert_to_lowercase(string)
print(lowercase_string)  # Output: "hello! this is a test string."
```

### Removing whitespaces
```python
from text_manipulation import remove_whitespaces

string = "   Hello!   "
trimmed_string = remove_whitespaces(string)
print(trimmed_string)  # Output: "Hello!"
```

### Extracting email addresses
```python
from text_manipulation import extract_emails

text = "Please contact us at info@example.com for more information."
emails = extract_emails(text)
print(emails)  # Output: ["info@example.com"]
```

### Calculating string similarity using cosine similarity
```python
from text_manipulation import calculate_cosine_similarity

string1 = "Hello, how are you?"
string2 = "Hi, how are you doing?"
similarity_score = calculate_cosine_similarity(string1, string2)
print(similarity_score)  # Output: 0.89
```

These examples showcase some of the key functionalities provided by this package. For a full list of available functions and their usage, please refer to the function definitions in the code.
```