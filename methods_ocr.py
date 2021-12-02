import os 
import numpy as np 
import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import datetime
import re
import difflib
import json
import nepali_roman as nr
import string
import streamlit as st


#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_info_front(image):
    text = pytesseract.image_to_string(image, lang ='nep', output_type=Output.DICT)
    return text['text']


def get_citizenship_number(result):
  lines=[]
  for line in result.split('\n'):
    lines.append(line)
  
  possible_list = []
  number = ''
  for line in lines:
    parts = line.split(' ')
    for letters in parts:
      num = letters.split('-')
      if len(num) != 1:
        possible_list.append(num)

  for i in possible_list:
    i = ''.join(i)
    try:
      i = int(i)
      return i
    except:
      pass


def get_gender(data):
  try: 
    my_str = ['purus', 'mahila']
    str_list = word_tokenize(data)
    best_match_male = difflib.get_close_matches(my_str[0],str_list,1)[0]
    score_male = difflib.SequenceMatcher(None, my_str[0], best_match_male).ratio()
    best_match_female = difflib.get_close_matches(my_str[1],str_list,1)[0]
    score_female = difflib.SequenceMatcher(None, my_str[1], best_match_female).ratio()
    if score_male > score_female:
      return 'male'
    else:
      return 'female'
  except:
    return None



def get_dob(date):
  try:
    for line in date.split('\n'):
      words = word_tokenize(line)
      for word in words:
        try:
          int(word)
          year_line = line
          break
        except:
          pass
    date = ''
    for word in year_line.split(' '):
      try:
        numeric = int(word)
        date+=str(numeric)+'/'
      except:
        pass
    if len(date)>8:
      return date[:-1]
    else:
      return None
  except:
    return None



def thresh_manual(image, param):
    image = cv2.threshold(image, param, 255, cv2.THRESH_BINARY)[1]
    return image



def word_tokenize_nep(sentence, new_punctuation=[]):

  punctuations = ['।', ',', ';', '?', '!', '—', '-', '.', ':', '॥', '|']
  if new_punctuation:
      punctuations = set(punctuations + new_punctuation)

  for punct in punctuations:
      sentence = ' '.join(sentence.split(punct))

  return sentence.split()



def get_name_both(data_nep):
  try: 
    my_str = ['thara', 'linga', 'थर', 'लिङ्ग']
    str_list_nep = word_tokenize_nep(data_nep)    
    best_match_nep_1 = difflib.get_close_matches(my_str[2],str_list_nep,1)[0]
    best_match_nep_2 = difflib.get_close_matches(my_str[3],str_list_nep,1)[0]
    ix3, ix4 = str_list_nep.index(best_match_nep_1), str_list_nep.index(best_match_nep_2)
    full_name_nep = str_list_nep[ix3+1:ix4]
    full_name_nep = ' '.join(full_name_nep)
    full_name = nr.romanize_text(full_name_nep).title()
 
    return full_name, full_name_nep
  except:
    return None, None



def sentence_tokenize_nep(text):
  sentences = text.strip().split(u"\n")
  sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
  return sentences


def get_duplicates_index(lst, item):
  return [i for i, x in enumerate(lst) if x == item]



def get_district(data):
  districts = ['भक्तपुर', 'चितवन', 'धादिङ', 'दोलखा', 'काभ्रे', 'काठमाडौं', 'ललितपुर', 'मकवानपुर', 'नुवाकोट',
             'रामेछाप', 'रसुवा', 'सिन्धुली', 'सिन्धुपाल्चोक', 'बाग्लुङ', 'गोरखा', 'कास्की', 'लमजुङ', 'मनाङ', 'मुस्ताङ', 'म्याग्दी',
             'नवलपरासी', 'पर्वत', 'स्याङ्जा', 'तनहुँ', 'बारा', 'धनुषा', 'महोत्तरी', 'पर्सा', 'रौतहट', 'सप्तरी', 'सर्लाही', 'सिराहा', 'दैलेख',
             'डोल्पा', 'हुम्ला', 'जाजरकोट', 'जुम्ला', 'कालिकोट', 'मुगु', 'रूकुम', 'सल्यान', 'सुर्खेत', 'भोजपुर', 'धनकुटा', 'इलाम', 'झापा',
             'खोटाङ', 'मोरङ', 'ओखलढुङ्गा', 'पाँचथर', 'संखुवासभा', 'सोलुखुम्बु', 'सुनसरी', 'ताप्लेजुङ', 'तेह्रथुम', 'उदयपुर', 'अर्घाखाँची',
             'बाँके', 'बर्दिया', 'दाङ', 'गुल्मी', 'कपिलवस्तु', 'नवलपरासी', 'पाल्पा', 'प्युठान', 'रोल्पा', 'रूकुम', 'रूपन्देही', 'अछाम', 'बैतडी',
             'बझाङ', 'बाजुरा', 'डडेलधुरा', 'दार्चुला', 'डोटी', 'कैलाली', 'कञ्‍चनपुर']
  try:
    data_tokenized = word_tokenize_nep(data)
    district_index = get_duplicates_index(data_tokenized, 'जिल्ला')
    best_match_birth_district = difflib.get_close_matches(data_tokenized[district_index[0]+1],districts,1)[0]
    best_match_permanent_district = difflib.get_close_matches(data_tokenized[district_index[1]+1],districts,1)[0]
    return best_match_birth_district, best_match_permanent_district
  except:
    return None, None




def get_ward_no(data):
  ward = []
  data_tokenized = word_tokenize_nep(data)
  try:
    match_1, match_2 = tuple(difflib.get_close_matches('वडा', data_tokenized, n=2))
    for line in data.split('\n'):
      tokenized_line = word_tokenize_nep(line)
      if match_1 in tokenized_line or match_2 in tokenized_line:
        try:
          word = tokenized_line[-1]
          word = int(word)
          ward.append(word)
        except:
          pass
    if len(ward) == 2:
      return tuple(ward)
    else:
      return None, None
  except:
    return None, None


    
def get_info_citizenship_2(image, threshold, typ):
  image = thresh_manual(image, threshold)
  result = get_info_front(image)
  result_romanized = nr.romanize_text(result)
  citizenship_number = get_citizenship_number(result_romanized)
  name, name_nep = get_name_both(result)
  gender = get_gender(result_romanized)
  dob = get_dob(result_romanized)
  birth_district, permanent_district = get_district(result)
  birth_ward, permanent_ward = get_ward_no(result)
  if typ == 'json':
    return json.dumps({'Citizenship number': citizenship_number,
                     'Name': name, 'Name (Nepali)': name_nep, 'Gender': gender, 
                     'Birth Date': dob, 'Birth Place': birth_district,
                     'Permanent Address': permanent_district,
                     'Birth Ward Number': birth_ward, "Permanent Ward Number": permanent_ward}, ensure_ascii=False)
  else:
    return citizenship_number, name, name_nep, gender, dob, birth_district, permanent_district, birth_ward, permanent_ward



def most_frequent(List):
  try:
    return max(set(List), key = List.count)
  except:
    return None



def get_accurate_info(image, start=100, stop=150, step_size=3, just_citizenship_num = True):
  citizenship_number_list = []
  name_list = []
  name_nep_list = []
  gender_list = []
  dob_list = []
  name_nep_list = []
  birth_district_list = []
  permanent_district_list = []
  birth_ward_list = []
  permanent_ward_list = []
  for i in range(start, stop+1, step_size):
    citizenship_number, name, name_nep, gender, dob, birth_district, permanent_district, birth_ward, permanent_ward = get_info_citizenship_2(image, i, 'tuple')
    if citizenship_number is not None:
      citizenship_number_list.append(citizenship_number)
    if name is not None:
      name_list.append(name)
    if name_nep is not None:
      name_nep_list.append(name_nep)
    if gender is not None:
      gender_list.append(gender)
    if dob is not None:
      dob_list.append(dob)
    if birth_district is not None:
      birth_district_list.append(birth_district)
    if permanent_district is not None:
      permanent_district_list.append(permanent_district)
    if birth_ward is not None:
      birth_ward_list.append(birth_ward)
    if permanent_ward is not None:
      permanent_ward_list.append(permanent_ward)
  

  accurate_citizenship_number = most_frequent(citizenship_number_list)
  accurate_name = most_frequent(name_list)
  accurate_name_nep = most_frequent(name_nep_list)
  accurate_gender = most_frequent(gender_list)
  accurate_dob = most_frequent(dob_list)
  accurate_birth_district = most_frequent(birth_district_list)
  accurate_permanent_district = most_frequent(permanent_district_list)
  accurate_birth_ward = most_frequent(birth_ward_list)
  accurate_permanent_ward = most_frequent(permanent_ward_list)
  
  if not just_citizenship_num:
    return json.dumps({'Citizenship number': accurate_citizenship_number,
                       'Name': accurate_name, 'Name (Nepali)': accurate_name_nep,'Gender': accurate_gender, 
                       'Birth Date': accurate_dob, 'Birth Place': accurate_birth_district,
                       'Permanent Address': accurate_permanent_district,
                       'Birth Ward Number': accurate_birth_ward, 'Permanent Ward Number': accurate_permanent_ward}, ensure_ascii=False)
  else:
    return accurate_citizenship_number


if __name__ == '__main__':
  image = cv2.imread(r"C:\Users\aarya\Downloads\aaryan_front.jpg")
  print(get_accurate_info(image))