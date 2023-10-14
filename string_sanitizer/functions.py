e
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def remove_special_characters(string):
    sanitized_string = re.sub(r"[^\w\s]", "", string)
    return sanitized_string

def remove_whitespaces(string):
    return string.strip()

def convert_to_lowercase(string):
    return string.lower()

def remove_digits(string):
    return ''.join(filter(lambda x: not x.isdigit(), string))

def remove_punctuation(string):
    return ''.join(char for char in string if char not in string.punctuation)

def replace_multiple_whitespaces(string):
    return re.sub(r'\s+', ' ', string)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def split_string_into_words(string):
    return string.split()

def join_words(words):
    return ' '.join(words)

def replace_words(string, replacements):
    for word, replacement in replacements.items():
        string = string.replace(word, replacement)
    
    return string

def sanitize_string(string):
    sanitized_string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
    sanitized_string = sanitized_string.lower()
    return sanitized_string

def tokenize_string(string, delimiters):
    tokens = []
    current_token = ""

    for char in string:
        if char in delimiters:
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char

    if current_token:
        tokens.append(current_token)

    return tokens

def standardize_string(string, format_rules=None):
    string = string.strip()
    
    string = string.title()
    
    if format_rules:
        for rule in format_rules:
            if rule == 'uppercase':
                string = string.upper()
            elif rule == 'lowercase':
                string = string.lower()
            elif rule == 'capitalize':
                string = string.capitalize()

    return string

def extract_email_addresses(string):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(pattern, string)
    return matches

def extract_urls(string):
    url_pattern = r"(https?://[^\s]+)"
    urls = re.findall(url_pattern, string)
    return urls

def lemmatize_string(input_string):
    lemmatizer = WordNetLemmatizer()
    words = input_string.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_string = ' '.join(lemmatized_words)
    return lemmatized_string

def count_word_frequency(string):
    word_list = string.split()
    word_count = {}
    
    for word in word_list:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    return word_count

def concatenate_strings(*args, sep=' '):
    return sep.join(args)

def replace_words(string, replacements):
    replacements = sorted(replacements, key=len, reverse=True)
    
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in replacements) + r')\b')
    
    result = pattern.sub(lambda x: replacements[x.group()], string)
    
    return resul