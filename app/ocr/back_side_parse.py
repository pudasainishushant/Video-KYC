from typing import NoReturn
import cv2
import os 
import numpy as np 
import pytesseract
from pytesseract import Output

import datetime
import re

import json
import difflib

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
districts = json.load(open(my_absolute_dirpath + "/tessdata/districts.txt")).values()
districts = [d.lower() for d in districts]

mInitial = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

def get_string_similarity(text1, text2):
    """Check string similarity

    Args:
        text1 (str): first string 
        text2 (str): second string

    Returns:
        int: similarity ratio
    """
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def convert_date_information(text):
    """ Date information conversion

    Args:
        text (str): input date str

    Returns:
        str: date format
    """
    dob = None

    year_and_day = [''.join(re.findall(re.compile(r"[0-9]"), y)) for y in text]
    year = [y for y in year_and_day if len(y) >= 4]
    if year:
        year = year[0]

    day = [y for y in year_and_day if 1<= len(y) <= 2]
    if day:
        day = day[0]

    # month =[''.join(dict.fromkeys(re.findall(re.compile(f"{[','.join(mInitial)]}"), y))) for y in text]
    month = [re.findall(r'[a-z]?:?([A-Z]{3})', n) for n in text]
    month = list(filter(None, month))
    if month and year and day:
        month = month[0]
        month = [m for m in month if m in mInitial]
        month = datetime.datetime.strptime(month[0], "%b").month

        dob = datetime.datetime(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    return dob

def get_Gender(text):
    """ Returns gender from input string

    Args:
        text (str): input string

    Returns:
        str: gender type
    """
    gender = None
    if get_string_similarity('Male', text[-1]) >= 0.75:
        gender = "Male"
    elif get_string_similarity('Female', text[-1]) >= 0.75:
        gender = "Female"
    elif get_string_similarity('Third', text[-1]) >= 0.75:
        gender = "Third"
    return gender

def get_addr_type_name(text):
    """ Returns address name and type

    Args:
        text (str): input string

    Returns:
        str: address type and name 
    """
    addr_type = None
    addr_name = None
    if get_string_similarity('Municipality', text[0]) > 0.75:
        addr_type = "Municipality"
    elif get_string_similarity('Sub-Metropolitan', text[0]) > 0.75:
        addr_type = "Sub-Metropolitan"  
    elif get_string_similarity('Metropolitan', text[0]) > 0.75:
        addr_type = "Metropolitan"
    elif get_string_similarity('VDC', text[0]) > 0.75:
        addr_type = "VDC"

    if addr_type:
        try:
            addr_name = text[text.index(text[0])+1]
        except:
            pass

    return addr_type, addr_name

def get_citizenship_number(text):
    """ Returns citizenship number and gender

    Args:
        text (str): input string

    Returns:
        str: citizenship number
        str: gender
    """
    cn = None
    gender = None
    i_dict = {}
    for i in text:
        il = ' '.join(c for c in i)
        r = get_string_similarity('Citizenship Certificate No. Sex Male Female Third', il)
        if r >= 0.5:
            i_dict[r] = i
    
    if len(i_dict) != 0:
        cnText = i_dict[max(i_dict)] # finding values from dict with highest key
        cn = [c for c in cnText if re.search("[0-9]", c)]
        gender = get_Gender(cnText)

    if cn is None:
        j_dict = {}
        for i in text:
            l = 0
            cn = [c for c in i if re.findall(r'\b([0-9_?]+)\b', c)]
            if cn:
                for j in cn:
                    l += len(j)
                j_dict[l] = cn

        if len(j_dict) != 0:
            cn = j_dict[max(j_dict)]

    if gender is None:
        for i in text:
            gl = [g for g in i if re.findall(r'Sex', g)]
            if gl:
                gender = get_Gender(i)

    if cn:
        cn = '-'.join(cn)
        cn = re.sub(r"^\W\s+|_","", cn)

    return cn, gender

def get_full_name(text):
    """ Returns full name

    Args:
        text (str): input string

    Returns:
        str: name
    """
    name = None
    for i in text:
        il = ' '.join(c for c in i)

        if re.findall(re.compile(r'Full Name'),il):
            try:
                i.remove('.')
            except:
                pass
            name = ' '.join(i[2:])

    if name is None:
        for i in text:
            fullName = [n for n in i if re.search(r'\b\n|^[A-Z\s]+\b', n)]
            if len(fullName) >= 2:
                name = ' '.join(fullName)
                break
    return name    

def get_DOB(text):
    """Returns Date Of Birth(DOB)

    Args:
        text (str): input string

    Returns:
        str: DOB
    """
    dob = None
    for i in text:
        il = ' '.join(c for c in i)
        if re.findall(re.compile(r'Date of Birth'), il):
            dob = convert_date_information(i)
            if dob:
                break

    if dob is None:
        for i in text:
            DOB = [re.findall(r'[a-z]:?([A-Z]{3})', n) for n in i]
            DOB = list(filter(None, DOB))
            if len(DOB) == 1 and DOB[0][0] in mInitial:
                dob = convert_date_information(i)
    return dob

def get_address(text):
    """Returns birth place and permanent address

    Args:
        text (str): input string

    Returns:
        str: birth place and permanent address
    """
    birth_place = {}
    permanent_addr = {}

    for i in text:
        il = ' '.join(c for c in i)
        
        # birth place
        r1 = get_string_similarity('Birth Place District', il)
        if r1 >= 0.4:
            district = [d.lower() for d in i]
            district = [d for d in district if d in districts]
            if district:
                birth_place['district'] = district[0]

            addr1 = text[text.index(i)+1]
            if addr1:
                addr_type, addr_name = get_addr_type_name(addr1)
                if addr_name is not None:
                    birth_place[addr_type] = addr_name

                ward = [re.findall(r'[0-9]+', c) for c in addr1]
                if ward:
                    ward = list(filter(None, ward))
                    if ward:
                        birth_place['ward'] = ward[0][0]

        # permanent address
        r2 = get_string_similarity('Permanent Address District', il)
        if r2 >= 0.5:
            district = [d.lower() for d in i]
            district = [d for d in district if d in districts]
            if district:
                permanent_addr['district'] = district[0]
            
            addr2 = text[text.index(i)+1]
            if addr2:
                addr_type, addr_name = get_addr_type_name(addr2)
                if addr_name is not None:
                    permanent_addr[addr_type] = addr_name

                ward = [re.findall(r'[0-9]+', c) for c in addr1]
                if ward:
                    ward = list(filter(None, ward))
                    if ward:
                        permanent_addr['ward'] = ward[0][0]

    return birth_place, permanent_addr

def get_issue_date(image):
    """Returns issue date

    Args:
        image (ndarray): input image

    Returns:
        str: issue date
    """
    issue_date = None
    os.environ["TESSDATA_PREFIX"] = my_absolute_dirpath + "/tessdata"
    h,w = image.shape
    image = image[h//2 : h, 0:w]
    text = pytesseract.image_to_data(image, lang ='nep', output_type=Output.DICT)
    text = text.get('text')
    text = [line for line in text if line.strip() != '']
    for i in text:
        if len(i) >= 8:
            il = i.split('-')
            if len(il) == 3:
                issue_date = i
    
    return issue_date

def get_info_back(image):
    """Returns all the information from input image

    Args:
        image (ndarray): input image

    Returns:
        dic: information from the input image
    """
    # os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(image, lang='eng')
    text = [line for line in text.split('\n') if line.strip() != '']

    info = {}

    # cnText = [i.split(' ') for i in text]
    cnText = []
    for i in text:
        ct = i.split(' ')
        ct = [re.sub("[$&+,:;=?@#|'<>.^*()%!-]", '', c) for c in ct]
        ct = list(filter(None, ct))
        cnText.append(ct)
    
    print(cnText)
    c_number, gender = get_citizenship_number(cnText)
    info['Citizenship Number'] = c_number
    info['Gender'] = gender
    
    full_name = get_full_name(cnText)
    info['Name'] = full_name

    date_of_birth = get_DOB(cnText)
    info['dob'] = date_of_birth

    birth_place, permanent_adrr = get_address(cnText)
    info['birth_place'] = birth_place
    info['permanent address'] = permanent_adrr

    issue_date = get_issue_date(image)
    info['issue_date'] = issue_date
    return info

def rotate_image(image, angle):
    """Image rotation

    Args:
        image (ndarray): input image
        angle (int): rotation angle

    Returns:
        ndarray: rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

if __name__ == '__main__':
    image = cv2.imread("../test/back2.jpg", 1)
    image = rotate_image(image, 2)
    print("Information Extracted:", get_info_back(image))

    cv2.imshow("image", image)
    cv2.waitKey()
