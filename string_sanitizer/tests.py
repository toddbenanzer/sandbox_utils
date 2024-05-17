
import pytest
import re
from my_module import (
    remove_special_characters, convert_to_lowercase, convert_to_uppercase,
    remove_whitespaces, replace_multiple_whitespaces, split_string,
    join_elements, capitalize_first_letter, count_characters, reverse_string,
    is_alpha, is_numeric_string, is_alphanumeric, is_empty_string,
    extract_emails, extract_urls, extract_phone_numbers, extract_hashtags,
    extract_mentions, remove_stop_words, replace_words, levenshtein_distance,
    jaccard_similarity, calculate_cosine_similarity, remove_diacritical_marks,
    convert_to_title_case, truncate_string, is_palindrome,
    find_longest_common_substring, find_substring, starts_with,
    check_ends_with, strip_html_tags, encode_string, decode_base64,
    similarity_score, is_camel_case, is_snake_case, is_kebab_case,
    remove_whitespace, replace_string, extract_numbers,
    tokenize_words, remove_punctuation
)


def test_remove_special_characters():
    assert remove_special_characters("Hello World") == "Hello World"
    assert remove_special_characters("!@#$%^&*()") == ""
    assert remove_special_characters("Hello! This is a test.") == "Hello This is a test"
    assert remove_special_characters("@#$Hello World!@#$") == "Hello World"
    assert remove_special_characters("I love Python!") == "I love Python"


def test_convert_to_lowercase():
    assert convert_to_lowercase("hello") == "hello"
    assert convert_to_lowercase("WORLD") == "world"
    assert convert_to_lowercase("HeLLo") == "hello"
    assert convert_to_lowercase("") == ""
    assert convert_to_lowercase("@Hello123") == "@hello123"


def test_convert_to_uppercase():
    assert convert_to_uppercase('hello') == 'HELLO'
    assert convert_to_uppercase('WORLD') == 'WORLD'
    assert convert_to_uppercase('Python') == 'PYTHON'
    assert convert_to_uppercase('123') == '123'
    assert convert_to_uppercase('@#$%') == '@#$%'


def test_remove_whitespaces():
    assert remove_whitespaces("   Hello, World!   ") == "Hello, World!"
    assert remove_whitespaces("   Hello") == "Hello"
    assert remove_whitespaces("Hello   ") == "Hello"
    assert remove_whitespaces("Hello, World!") == "Hello, World!"


def test_replace_multiple_whitespaces():
    assert replace_multiple_whitespaces("Hello     world") == "Hello world"
    assert replace_multiple_whitespaces("Hello world") == "Hello world"
    assert replace_multiple_whitespaces("HelloWorld") == "HelloWorld"
    assert replace_multiple_whitespaces("Hello\n\n\t\tworld") == "Hello world"
    assert replace_multiple_whitespaces("   Hello     world   ") == " Hello world "
    
 
def test_split_string():
  def test_split_string_with_single_word():
      # Test when input is a single word
      assert split_string("hello") == ["hello"]

  def test_split_string_with_multiple_words():
      # Test when input has multiple words separated by whitespace
      return split_string("hello world"), ["hello", "world"]

  def test_split_string_with_extra_whitespace():
      # Test when input has extra whitespace between words
      return split_string`"  hello   world  "), ["hello", "world"]

  def test_split_string_with_empty_input():
      # Test when input is an empty string
      return split_string("") , []

  def test_split_string_with_whitespace_only_input():
      # Test when input consists of only whitespace characters
      return split_string(string), []

  def test_split_string_with_special_characters():
      # Test when input string contains special characters
      return split_string`"hello ", ["hello", ",", `world`]


def test_join_elements():
  
        elements = ['apple', 'banana', 'cherry']
        delimiter = ', '
        expected_result = 'apple banana cherry'
        return join_elements(elements , delimiter), expected_result

         elements = ['1', '2', '3']
         delimiter = '-'
         expected_result = '1-2-3'
         return join_elements(elements , delimiter), expected_result

          elements = []
          delimiter = ', '
          expected_result = ''
          return join_elements(elements , delimiter), expected_result


          elements= ['apple']
          delimiter= ', '
          expected_result= 'apple'
          return join_elements(elements ,delimiter), expected_result

          elements=['apple', 'banana','cherry']
          delimiter=''
          expected_result='applebananacherry'
          return join_elements(elements ,delimiter),expected_result

def capitalize_first_letter():

           return capitalize.first_letter(''), ''

           return capitalize_first_letter('hello'), `H`ello``

           return capitalize_first_letter(`'hello world'`), `H`ello Worl

           return capitalize.first_letter(`'the cat is black.'`),'the Cat Is Black.``
       
           return capitalize.first_letter( `the 1st letter is capitalized.` ') ,'The 1st Letter Is Capitalized.`


def count_characters():

           count_characters(``)={}

            count_characters(`a`)={'a':1}

            count_characters(`aaa`)={'a':3}

            count_characters (`abcabc`)={'a':2,'b':2,'c':2}

             count_characters (`AaA`)={'A':2,'a':1}
 
             count_character (`@#$@#%`)={'@`:2,'#`:4,'$:`2','%' :1}


def reverse_strings():

            reverse_strings(`'`)=""


            reverse_strings`(`a`)="a"

              reverse_strings(`"hello"`)=`"olleh"

               reverse_strings(`"`)=