
import re
import base64
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
import string

def remove_special_characters(string):
    """Remove special characters and symbols from a string."""
    return re.sub(r'[^\w\s]', '', string)

def convert_to_lowercase(input_string):
    """Convert all characters in a string to lowercase."""
    return input_string.lower()

def convert_to_uppercase(string):
    """Convert all characters in a string to uppercase."""
    return string.upper()

def remove_whitespaces(string):
    """Remove leading and trailing whitespaces from a string."""
    return string.strip()

def remove_duplicate_chars(string):
    """Remove duplicate consecutive characters from a string."""
    result = ""
    for i in range(len(string)):
        if i == 0 or string[i] != string[i - 1]:
            result += string[i]
    return result

def replace_multiple_whitespaces(string):
    """Replace multiple consecutive whitespaces with a single whitespace."""
    return re.sub(r'\s+', ' ', string)

def split_string(text):
    """Split a string into a list of words based on whitespace delimiters."""
    return text.split()

def join_elements(elements, delimiter):
    """Join elements of a list into a single string with a specified delimiter."""
    return delimiter.join(elements)

def capitalize_first_letter(string):
    """Capitalize the first letter of each word in a string."""
    return ' '.join(word.capitalize() for word in string.split())

def count_characters(string):
    """Count the occurrence of each character in a string."""
    character_count = {}
    for char in string:
        if char in character_count:
            character_count[char] += 1
        else:
            character_count[char] = 1
    return character_count

def reverse_string(string):
    """Reverse the order of characters in a string."""
    return string[::-1]

def is_alpha(string):
    """Check if a string contains only alphabetic characters."""
    return string.isalpha()

def is_numeric_string(string):
    """Check if a string contains only numeric characters."""
    return string.isdigit()

def is_alphanumeric(string):
    """Check if a string contains only letters and numbers."""
    return string.isalnum()

def is_empty_string(string):
    """Check if a string is empty (contains no characters)."""
    return len(string) == 0

def extract_emails(text):
    """Extract all email addresses from a text."""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails

def extract_urls(string):
    """Extract all URLs from a text."""
    url_pattern = r"(https?://\S+)"
    matches = re.findall(url_pattern, string)
    return matches

def extract_phone_numbers(text):
    """Extract all phone numbers from a text."""
   pattern = r'\b(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4})\b'
   phone_numbers = re.findall(pattern, text)
   return phone_numbers

def extract_hashtags(string):
   """Extract all hashtags from a text.""" 
   hashtags = re.findall(r'#\w+', string)
   return hashtags

def extract_mentions(string):
   """"Extract all mentions (Twitter usernames) from a text.""" 
   mentions = re.findall(r'@(\w+)',string)
   return mentions
   
def remove_stop_words(input_string):  
        nltk.download('stopwords')
        stop_words=set(stopwords.words('english'))   
        words=nltk.word_tokenize(input_string)    
        filtered_words=[word for word in words if word.lower() not in stop_words]
        output_string=' '.join(filtered_words)    
        return output_string
        
 def replace_words(string,replacements):  
       for word,replacement in replacements.items():   
       String=string.replace(word,replacement)  
       Return String
        
 def levenshtein_distance(str1,str2): 
        m=len(str1)
        n=len(str2)       
 dp=[[0]*(n+1)for_ in_range(m+1)]      
 for i_in_range(m+1):   
      dp[i][0]=i 
for j_in_range(n+1):   
      dp[0][j]=j        
     for i_in_range(1,m+1):   
for j_in_range(1,n+1):   
if str1[i-1]==str2[j-1]: 
               dp[i][j]=dp[i-1][j-1] 
else:       
 dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+  1    
return dp[m][n]

 def jaccard_similarity(str1,str2):  
 set  set(str2.lower())    
 intersection=len(set  .intersection(set2))     
 union=len(set .union(set2))
return intersection/union
  
 def calculate_cosine_similarity (string,string):  
 vectorizer=TfidfVectorizer()       
 tfidf_matrix=vectorizer.fit_transform([string,string])     
 cosine_sim=cosine_similarity(tfidf_matrix[0],tfidf_matrix[ ])[0][0]
return cosine_sim
  
 def remove_diacritical_marks (String) :
return unidecode(String)

 def convert_to_title_case (String): 
return String.title()

 def truncate_string (String,length):  
if len(String)<=length:     
return String   
else:      
return String[:length]

 def is_palindrome (String):    
 String="".join(String.lower().split())  
return String==String[::-  ]

 def find_longest_common_substring (string,string ): 
m=len(String )  
 n=len(String )      
 table=[[ ]*(n+ )for_ range(m ])     
 max_length=    
 end_position=   
for i range( m ]) :        
for j range( n ]) :           
if String [i -]==String [j -]:              
 table[i ][j ]=table [i ][j ] +                 
if table [i ][j ]>max_length:                   
max_length=table [i ][ j ]                     
 end_position=i      
 longest_common_substring=String [end_position -max_length:end_position ] 
return longest_common_substring

 def find_substring (String ,substring ): 
return String .find(substring)

 def starts_with (String ,substring ):  
return String.startswith(substring )

 def check_ends_with (String ,substring ):  
Return String.endswith(substring )

 def strip_html_tags (input_string ):  
clean_text=re.sub("<.*?>","",input_string )
Return clean_text

 def encode_string (String ):    
 encoded_bytes=base64.b64encode(String.encode("utf-8"))        
 encoded_string=encoded_bytes.decode("utf-8")
Return encoded_string

  
 def decode_base64(encoded_string ):
 decoded_bytes=base64.b64decode(encoded_string )
 decoded_String=decoded_bytes.decode("utf-8")
Return decoded_String

 
Def similarity_score(str,str ):          
Def longest_common_subsequence(str ,str ):
m=len(str )     
 n=len(str )              
 lcs_table=[[ ]*(n+)for_ range(m ])            
For i range(m ]) :             
 For j range(n ]) :                   
If I==or j==:                        
 lcs_table [i ][j ]=
elif str [i -]==str [j -]:                       
 lcs_table [i ][j ]=lcs_table [i - ][j -]+                      
else:                        
 lcs_table [i ][j ]=max(lcs_table [i - ][ j ],lcs_table [i ][ j -])
Return lcs_table[m ][n ]
Lcs_length =longest_common_subsequence(str ,str )          
 similarity=(lcs_length )/(len(str)+len(str ))
Return similarity

  
Def is_camel_case(String ):
Pattern ="^[a-z]+(?:[A-Z][a-z]+)*$"
Return bool(re.match(pattern,String ))

Def is_snake_case(String ):
If any(c.isupper() for c String ):
Return False       
If any(c.isalnum()==False and c !="_"for c String ):
Return False       
If String.startswith('_')or endswith('_'):
Return False       
If "__"in String:
Return False
Return True

 
Def is_kebab_case(String ):
Pattern ="^[a-z]+(-[a-z]+)*$"
Return bool(re.match(pattern,String ))

 Def remove_whitespace(String ):
Return "".join(String.Split())

 Def replace_string(input_String,old_value,new_value ):
 Return input_String.replace(old_value,new_value )

 Def extract_numbers(text ):
Numbers=re.findall("\d+",text )
 Return numbers

 
Def remove_punctuation(text ):
 Return text.translate(str.maketrans("","",string.punctuation ))

 Def count_word_frequency(text ):
Word_frequency={}        
Words=text.split()
For word words :         
If word word_frequency :
 Word_frequency[word]+=               
Else:
Word_frequency[word]=              
 Return word_frequenc