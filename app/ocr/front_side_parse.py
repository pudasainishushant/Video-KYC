import os
import numpy as np 
import datetime
import cv2
import pytesseract
from pytesseract import Output
from nltk.tokenize import word_tokenize
import re
import difflib
import nepali_roman as nr
import string
import pandas as pd

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

def get_info_front(image):
    os.environ["TESSDATA_PREFIX"] = my_absolute_dirpath + "/tessdata"

    # text = pytesseract.image_to_string(image, lang ='nep',config='-c preserve_interword_spaces=1', output_type=Output.DICT)
    # return text['text']

    custom_config = r'-c preserve_interword_spaces=1'
    d = pytesseract.image_to_data(image, lang = 'nep', config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)
    # clean up blanks
    df1 = df[(df.conf!='-1')&(df.text!=' ')&(df.text!='')]
    # sort blocks vertically
    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    text = ''
    for block in sorted_blocks:
        curr = df1[df1['block_num']==block]
        sel = curr[curr.text.str.len()>3]
        char_w = (sel.width/sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        for ix, ln in curr.iterrows():
            # add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0  # num of spaces that should be added
            if ln['left']/char_w > prev_left + 1:
                added = int((ln['left'])/char_w) - prev_left
                text += ' ' * added 
            text += ln['text'] + ' '
            prev_left += len(ln['text']) + added + 1
        text += '\n'
    return text
        
def get_similarity_score(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def get_citizenship_number(result, name_index=None):
    result_romanized = nr.romanize_text(result)

    cn_index = None
    numbers  = []
    lines=[]

    for line in result_romanized.split('\n'):
        lines.append(line)
    
    possible_list = {}
    for line in lines:
        digit_freq = re.findall('[0-9]+', line)
        possible_list[len(''.join(digit_freq))] = line
        
    if len(possible_list) != 0:
        cn = possible_list[max(possible_list)]
        cn_index = result_romanized.split('\n').index(cn)
    else:
        if name_index:
            cn = lines[name_index - 1]
    if cn:
        for i in cn:
            i = ''.join(i)
            try:
                j = int(i)
                numbers.append(str(j))
            except:
                pass
    if cn_index <= 10:
        return cn_index, ''.join(numbers)

    else:
        return None, None
    
def get_gender(data):
    gender = None
   
    my_str = ['purus', 'mahila']
    data = nr.romanize_text(data)
    str_list = word_tokenize(data)
    
    best_match_male = difflib.get_close_matches(my_str[0],str_list,1)
    best_match_female = difflib.get_close_matches(my_str[1],str_list,1)

    if best_match_male:
        score_male = difflib.SequenceMatcher(None, my_str[0], best_match_male[0]).ratio()
        if score_male >= 0.8:
            gender = 'Male'

    elif best_match_female:
        score_female = difflib.SequenceMatcher(None, my_str[1], best_match_female[0]).ratio()
        if score_female >= 0.8:
            gender = 'Female'
  
    return gender
  
def dob_conversion(text):
    year, month, day = None, None , None

    cnText = text.split()
    dob_list = []
    for c in cnText:
        try:
            dob_list.append(str(int(c)))
        except:
            pass
    
    if len(dob_list)>= 3:
        for d in dob_list:
            if len(d) == 4:
                year = d
        if year:
            y_index = dob_list.index(year)
            if len(dob_list[y_index+1:]) == 2:
                month = dob_list[dob_list.index(year)+1]
        if month:
            day = dob_list[dob_list.index(month)+1]

    if day is not None and int(day) <= 32 and int(month) <= 12:
        return datetime.datetime(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    else:
        return None

def get_dob(date, cn_index, name_index):
    dob= None
    i_dict = {}

    lines = date.split('\n')
    for line in lines:
        il = ''.join(c for c in line)
        il = word_tokenize_nep(il)
        r = difflib.SequenceMatcher(None, ' जन्म मिति:         साल:  २०५२  महिना  ०६  गते: १७ ', il).ratio()
        if r >= 0.5:
            line = nr.romanize_text(line)
            i_dict[r] = line
    
    if len(i_dict) != 0:
        cnText = i_dict[max(i_dict)] # finding values from dict with highest key
        dob = dob_conversion(cnText)
        
    if dob is None:
        if cn_index:
            cnText = lines[cn_index+6]
        elif name_index:
            cnText = lines[name_index+5]
        
        if cnText:
            dob = dob_conversion(cnText)

    return dob

def word_tokenize_nep(sentence):
    punc = '''!()-[]{};॥|:'"\, <>./?@#$%^&*_~'''

    for ele in sentence:
        if ele in punc:
            sentence = sentence.replace(ele, ' ')
    return sentence

def get_name_gender(data_nep, cn_index):
    name, gender, name_index = None, None, None
    possible_list = []
    my_str = ['नाम','थर', 'लिङ्ग']

    lines = data_nep.split('\n')
    for line in lines:
        if (re.findall(my_str[0], line) or re.findall(my_str[1], line)) or re.findall(my_str[2], line):
            if cn_index:
                if lines.index(line) == cn_index + 1:
                    possible_list.append(line)
            else:
                possible_list.append(line)

    if len(possible_list) == 0:
        for line in lines:
            if (re.findall(my_str[0], line) or re.findall(my_str[1], line)) or re.findall(my_str[2], line):
                possible_list.append(line)
    
    if len(possible_list) != 0:
        nl = possible_list[0]
        nameL = [s.strip() for s in nl.split('   ') if s]
        if len(nameL)>= 3:
            gender = get_gender(nameL[-1])
            if gender:
                nameL = nameL[:-1]
            else:
                nameL = nameL[:-1]
        else:
            gender = get_gender(nameL[0])
            if gender:
                nameL = nameL[:-1]

        nameL = word_tokenize_nep(' '.join(nameL)).split()
        try:
            surname_idx_1 = nameL.index(my_str[1])
            name = nameL[surname_idx_1+1:]
            name = ' '.join(name)          
        except:
            pass
        
    if name is None:
        try:
            surname_idx_2 = nameL.index(my_str[0])
            name = nameL[surname_idx_2+2:]
            name = ' '.join(name)
        except:
            pass
        
    if name is not None:
        for line in lines:
            raw_name = re.search(name, line)
            if raw_name and name_index is None:
                name_index = lines.index(line)
    return name, gender, name_index

def get_address(text, cn_index, name_index):
    birth_district, birth_ward= None, None
    permanent_district, permanent_ward = None, None

    lines = text.split('\n')
    # print(lines)
    if cn_index:
        b_dist_line = lines[cn_index+2]
        b_adds_line = lines[cn_index+3]
        p_dist_line = lines[cn_index+4]
        p_adds_line = lines[cn_index+5]
    elif name_index:
        b_dist_line = lines[name_index+1]
        b_adds_line = lines[name_index+2]
        p_dist_line = lines[name_index+3]
        p_adds_line = lines[name_index+4]

    b_dist_line = b_dist_line.split('   ')
    b_adds_line = b_adds_line.split('   ')
    p_dist_line = p_dist_line.split('   ')
    p_adds_line = p_adds_line.split('   ')
    # print(b_dist_line,"--------\n", b_adds_line,"--------\n", p_dist_line,"--------\n", p_adds_line)

    for i in b_dist_line:
        if len(i)>=2:
            rdis = difflib.SequenceMatcher(None, ' जिल्ला: ', i).ratio()
            # print(rdis, i)
            if rdis >= 0.65:
                birth_district = get_district(i)

    for i in p_dist_line:
        if len(i)>=2:
            rdis_p = difflib.SequenceMatcher(None, ' जिल्ला:', i).ratio()
            # print(rdis_p, i)
            if rdis_p >= 0.65:
                permanent_district = get_district(i)

    for i in b_adds_line:
        if len(i)>=2:
            rbward = difflib.SequenceMatcher(None, 'वडा  नं. :', i).ratio()
            if rbward>=0.65:
                romanized_i = nr.romanize_text(i)
                birth_ward= re.findall(r'[0-9]+', romanized_i)
                birth_ward = str(birth_ward[0])

    for i in p_adds_line:
        if len(i)>=2:
            rpward = difflib.SequenceMatcher(None, 'वडा  नं. :', i).ratio()
            if rpward>=0.65:
                romanized_i = nr.romanize_text(i)
                permanent_ward= re.findall(r'[0-9]+', romanized_i)
                permanent_ward = str(permanent_ward[0])

    return birth_district, birth_ward, permanent_district, permanent_ward

def get_parent_name(text, cn_index, name_index, name):
    father_name, mother_name = None, None

    lines = text.split('\n')
    # print(lines)
    if cn_index:
        fname_line = lines[cn_index+7]
        mname_line = lines[cn_index+9]
    elif name_index:
        fname_line = lines[name_index+6]
        mname_line = lines[name_index+8]

    # print(fname_line.split('   '), '::::', mname_line.split('   '))
    f_dict, m_dict = {}, {}
    f_dic_index, m_dic_index = 0,0
    fname_line = fname_line.split('   ')
    mname_line = mname_line.split('   ')

    for i in fname_line:
        if len(i)>=2:
            rf = difflib.SequenceMatcher(None, 'बाबुको नाम थर:', i).ratio()
            # print(rf, i)
            if rf >= 0.65:
                f_dict[rf] = i
                f_dic_index = fname_line.index(i)

    if f_dict !=0 and f_dic_index!=0:
        if len(fname_line[f_dic_index+1])>4:
            father_name = fname_line[f_dic_index+1]
        elif len(fname_line[f_dic_index+2])>4:
            father_name = fname_line[f_dic_index+2]
        else:
            father_name = None

        # print(fname_line[f_dic_index+1])

    for i in mname_line:
        if len(i)>=2:
            rm = difflib.SequenceMatcher(None, 'आमाको नाम थर:', i).ratio()
            # print(rm, i)
            if rm >= 0.65:
                m_dict[rm] = i
                m_dic_index = mname_line.index(i)

    if m_dict !=0 and m_dic_index!=0:
        # print(mname_line[m_dic_index+1])
        if len(mname_line[m_dic_index+1])>=4:
            mother_name = mname_line[m_dic_index+1]
        elif len(mname_line[m_dic_index+2])>=4:
            mother_name = mname_line[m_dic_index+2]
        else:
            mother_name = None


            # rm = difflib.SequenceMatcher(None, 'ammaak0 naama thara', i[0]).ratio()
            # # print(rm, ir[0])
            # if rm >= 0.7:
            #     m_dict[rm] = i
            
    # if len(f_dict) != 0:
    #     f_dict = sorted(f_dict.items(), reverse=True) # sorting dict with highest key  
        
    #     for n in f_dict:
    #         fn = n[1][1]
    #     f_dict = {}
    #     m_dict = {}
        
    #     for i in text:
    #         i = i.split('  ')

    #         i = list(filter(None, i))

    #         ir = [nr.romanize_text(j) for j in i]
        
    #         if len(ir)>=2:
    #             rf = difflib.SequenceMatcher(None, 'baabuk0 naama thara', ir[0]).ratio()
    #             # print(rf, ir[0])
    #             if rf >= 0.7:
    #                 f_dict[rf] = i
    #             rm = difflib.SequenceMatcher(None, 'ammaak0 naama thara', ir[0]).ratio()
    #             # print(rm, ir[0])
    #             if rm >= 0.7:
    #                 m_dict[rm] = i
            
    # if len(f_dict) != 0:
    #     f_dict = sorted(f_dict.items(), reverse=True) # sorting dict with highest key  
        
    #     for n in f_dict:
    #         fn = n[1][1]
    #         if fn != name: 
    #             father_name = fn
    #             break
    
    # if len(m_dict) != 0:
    #     m_dict = sorted(m_dict.items(), reverse=True) # sorting dict with highest key 
    #     # print(m_dict) 
    #     for n in m_dict:
    #         if len(n) >= 2:
    #             possible_name = n[1]
    #     m_dict = sorted(m_dict.items(), reverse=True) # sorting dict with highest key 
    #     # print(m_dict) 
    #     for n in m_dict:
    #         if len(n) >= 2:
    #             possible_name = n[1]
    #             for p in possible_name:
    #                 q = word_tokenize(p)
    #                 if 'थर' in q:
    #                     if q.index('थर') == len(q) - 1:
    #                         pmn = possible_name[1]
    #                         if pmn != name and pmn != father_name:
    #                             mother_name = pmn 
    #                     else:
    #                         pmn = ' '.join(q[q.index('थर')+1:])
    #                         if not mother_name and pmn != name and pmn != father_name:
    #                             mother_name = pmn
                    
                # if nr.romanize_text(mn) != name_nr and nr.romanize_text(mn) != nr.romanize_text(father_name): 
                #     mother_name = mn
                #     break
    # print(father_name, mother_name)
    return father_name, mother_name

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
        for d in districts:
            if re.findall(d, data):
                best_match_district = d

        # data_tokenized = word_tokenize_nep(data)
        # district_index = get_duplicates_index(data_tokenized, 'जिल्ला')
        # best_match_birth_district = difflib.get_close_matches(data_tokenized[district_index[0]+1],districts,1)[0]
        # best_match_permanent_district = difflib.get_close_matches(data_tokenized[district_index[1]+1],districts,1)[0]
        return best_match_district
        # return best_match_birth_district, best_match_permanent_district
    except:
        return None

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
    
def get_info_citizenship(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = get_info_front(image)
    result = ''.join(c for c in result)
    result = re.sub(r'\n\s*\n', '\n', result, flags=re.MULTILINE)
    # result = [ele for ele in result if ele.strip()]
    result = result.strip()
    print(result)
    dob, father_name, mother_name, birth_district, permanent_district, birth_ward, permanent_ward = None, None, None, None, None, None, None
    result_romanized = nr.romanize_text(result)
    cn_index, citizenship_number = get_citizenship_number(result)
    name, gender, name_index = get_name_gender(result, cn_index)
    if name_index and not cn_index:
        _ , citizenship_number = get_citizenship_number(result, name_index)

    if name_index or cn_index:
        dob = get_dob(result, cn_index, name_index)
        father_name, mother_name = get_parent_name(result, cn_index, name_index, name)
        birth_district, birth_ward, permanent_district, permanent_ward = get_address(result, cn_index, name_index)
    
    # permanent_district = None
    # permanent_district = get_district(result)
    # permanent_ward = get_ward_no(result)
    info = {'Citizenship number': citizenship_number,
                    'Name': name, 'Gender': gender, 
                    'Birth Date': dob,
                    'Father name' : father_name,
                    'Mother name' : mother_name,
                    'Birth Place': birth_district,
                    'Permanent Address': permanent_district,
                    'Birth Ward Number': birth_ward, "Permanent Ward Number": permanent_ward}
    return info

def most_frequent(List):
    try:
        return max(set(List), key = List.count)
    except:
        return None


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
    # i=2
    # while i<=10:
    #     image_path = "../test/front/front_{}.jpg".format(i)
    #     print(image_path)
    #     image = cv2.imread(image_path)
    #     # image = rotate_image(image, 0.5)
    #     print(get_info_citizenship(image))
    #     i+=1
    image_path = "../test/front/front_1.jpg"
    image = cv2.imread(image_path)
    # image = rotate_image(image, 0.5)
    print(get_info_citizenship(image))
    # final_res = {}
    # import time 
    # Range = [r for r in range(-3,4,0.5)]  
    
    # stime = time.time()
    # for i in Range:
    #     image_c= rotate_image(image, i)
    #     result = get_info_citizenship(image_c)

    #     for k, v in enumerate(result):
    #         if result[v] is not None:
    #             if len(result[v]) != 0:
    #                 final_res[v] = result[v]

    # print(get_info_citizenship(image))
    # print("final result: ", final_res)
