from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

import numpy as np
import os
import xml.etree.ElementTree as ET
import html
import HTMLParser
import re

stop = set(stopwords.words('english')) #- set('no')

# def _get_abs_pos(cur, ids):
#   min_dist = 1000
#   for i in ids:
#     if abs(cur - i) < min_dist:
#       min_dist = abs(cur - i)
#   if min_dist == 1000:
#     raise("[!] ids list is empty")
#   return min_dist

# def _count_pre_spaces(text):
#   count = 0
#   for i in xrange(len(text)):
#     if text[i].isspace():
#       count = count + 1
#     else:
#       break
#   return count

# def _count_mid_spaces(text, pos):
#   count = 0
#   for i in xrange(len(text) - pos):
#     if text[pos + i].isspace():
#       count = count + 1
#     else:
#       break
#   return count

# def _check_if_ranges_overlap(x1, x2, y1, y2):
#   return x1 <= y2 and y1 <= x2

# def _get_data_tuple(text, asp_term, fro, to, label, word2idx):
#   #words = word_tokenize(text)
#   words = text.split()
#   # Find the ids of aspect term
#   ids, st, i = [], _count_pre_spaces(text), 0
#   #print "Boom this was the text ", text, " Boom these are words ", words
#   for word in words:
#     if _check_if_ranges_overlap(st, st+len(word)-1, fro, to-1):
#       ids.append(i)
#     st = st + len(word) + _count_mid_spaces(text, st + len(word))
#     i = i + 1
#   pos_info, i = [], 0
#   for word in words:
#     #print word
#     pos_info.append(_get_abs_pos(i, ids))
#     i = i + 1
#   lab = None
#   if label == 'negative':
#     lab = 0
#   elif label == 'neutral':
#     lab = 1
#   elif label == 'positive':
#     lab = 2
#   else:
#     lab = 3
#   return pos_info, lab

# def clean_text(text):
#     text = text.rstrip()

#     if '""' in text:
#         if text[0] == text[-1] == '"':
#             text = text[1:-1]
#         text = text.replace('\\""', '"')
#         text = text.replace('""', '"')

#     text = text.replace('\\""', '"')

#     text = HTMLParser.HTMLParser().unescape(text)
#     text = ' '.join(text.split())
#     return text

def load_embedding_file(embed_file_name, word_set):
  ''' loads embedding file and returns a dictionary (word -> embedding) for the words existing in the word_set '''

  embeddings = {}
  with open(embed_file_name, 'r') as embed_file:
    for line in embed_file:
      content = line.strip().split()
      word = content[0]
      if word in word_set:
        embedding = np.array(content[1:], dtype=float)
        embeddings[word] = embedding

  return embeddings

def get_dataset_resources(data_file_name, sent_word2idx, target_word2idx, word_set, max_sent_len):
  ''' updates word2idx and word_set '''
  if len(sent_word2idx) == 0:
    sent_word2idx["<pad>"] = 0

  word_count = []
  sent_word_count = []
  target_count = []

  words = []
  sentence_words = []
  target_words = []

  with open(data_file_name, 'r') as data_file:
    lines = data_file.read().split('\n')
    for line_no in range(0, len(lines)-1, 3):
      sentence = lines[line_no]
      target = lines[line_no+1]

      sentence.replace("$T$", "")
      sentence = sentence.lower()
      sentence = re.sub(r'[\?.!]', ';', sentence)
      target = target.lower()
      max_sent_len = max(max_sent_len, len(sentence.split()))
      sentence_words.extend(sentence.split())
      target_words.extend([target])
      words.extend(sentence.split() + target.split())

    sent_word_count.extend(Counter(sentence_words).most_common())
    target_count.extend(Counter(target_words).most_common())
    word_count.extend(Counter(words).most_common())

    for word, _ in sent_word_count:
      if word not in sent_word2idx:
        sent_word2idx[word] = len(sent_word2idx)

    for target, _ in target_count:
      if target not in target_word2idx:
        target_word2idx[target] = len(target_word2idx)    

    for word, _ in word_count:
      if word not in word_set:
        word_set[word] = 1

  return max_sent_len

def get_embedding_matrix(embeddings, sent_word2idx,  target_word2idx, edim):
  ''' returns the word and target embedding matrix ''' 
  word_embed_matrix = np.zeros([len(sent_word2idx), edim], dtype = float)
  target_embed_matrix = np.zeros([len(target_word2idx), edim], dtype = float)

  for word in sent_word2idx:
    if word in embeddings:
      word_embed_matrix[sent_word2idx[word]] = embeddings[word]

  for target in target_word2idx:
    for word in target:
      if word in embeddings:
        target_embed_matrix[target_word2idx[target]] += embeddings[word]
    target_embed_matrix[target_word2idx[target]] /= max(1, len(target.split()))

  print type(word_embed_matrix)
  return word_embed_matrix, target_embed_matrix


def get_dataset(data_file_name, sent_word2idx, target_word2idx, embeddings):
  ''' returns the dataset'''
  sentence_list = []
  location_list = []
  target_list = []
  polarity_list = []

  original_sentence_list = []
  DeltaI_mm_list = []
  W_ma_list = []

  with open(data_file_name, 'r') as data_file:
    lines = data_file.read().split('\n')
    for line_no in range(0, len(lines)-1, 3):
      sentence = lines[line_no].lower()
      target = lines[line_no+1].lower()
      polarity = int(lines[line_no+2])

      sent_words = sentence.split()
      target_words = target.split()
      try:
        target_location = sent_words.index("$t$")
      except:
        print "sentence does not contain target element tag"
        exit()


      target_locations = range(target_location,target_location+len(target_words))
      original_sentence = sentence.replace("$t$", target)
      try:
        # DI_mm, W_ma =  np.zeros(2*[70]), np.zeros([70, 1])#Ctree.getReuiredParameters(sentence = original_sentence, aspect_words_indexes = target_locations)
        DI_mm, W_ma = Ctree.getReuiredParameters(sentence = original_sentence, aspect_words_indexes = target_locations)
        #
      except Exception as e:
        print e
        continue


      is_included_flag = 1
      id_tokenised_sentence = []
      location_tokenised_sentence = []
      
      for index, word in enumerate(sent_words):
        if word == "$t$":
          continue
        try:
          word_index = sent_word2idx[word]
        except:
          print "id not found for word in the sentence"
          exit()

        location_info = abs(index - target_location)

        if word in embeddings:
          id_tokenised_sentence.append(word_index)
          location_tokenised_sentence.append(location_info)

        # if word not in embeddings:
        #   is_included_flag = 0
        #   break

      is_included_flag = 0
      for word in target_words:
        if word in embeddings:
          is_included_flag = 1
          break
          

      try:
        target_index = target_word2idx[target]
      except:
        print target
        print "id not found for target"
        exit()


      if not is_included_flag:
        print sentence
        continue

      sentence_list.append(id_tokenised_sentence)
      location_list.append(location_tokenised_sentence)
      target_list.append(target_index)
      polarity_list.append(polarity)
      original_sentence_list.append(original_sentence)
      DeltaI_mm_list.append(DI_mm)
      W_ma_list.append(W_ma)

  return sentence_list, location_list, target_list, polarity_list, original_sentence_list, DeltaI_mm_list, W_ma_list




# def read_data(fname, source_count, source_word2idx, target_count, target_word2idx):
#   boobs = {0:0,1:0,2:0,3:0}
#   if os.path.isfile(fname) == False:
#     raise("[!] Data %s not found" % fname)

#   tree = ET.parse(fname)
#   root = tree.getroot()

#   source_words, target_words, max_sent_len = [], [], 0
#   for sentence in root:
#     text =  sentence.find('text').text.lower()
#     # text = clean_text(sentence.find('text').text.lower())
#     text = ' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', text).lower().split())
#     #print text
#     #source_words.extend(word_tokenize(text))
#     source_words.extend(text.split())
#     if len(word_tokenize(text)) > max_sent_len:
#       #max_sent_len = len(word_tokenize(text))
#       max_sent_len = len(text.split())
#     for asp_terms in sentence.iter('aspectTerms'):
#       for asp_term in asp_terms.findall('aspectTerm'):
#         target_words.append(asp_term.get('term').lower())
#   if len(source_count) == 0:
#     source_count.append(['<pad>', 0])
#     source_count.append(['$T$', 0])
#   source_count.extend(Counter(source_words).most_common())
#   target_count.extend(Counter(target_words).most_common())

#   for word, _ in source_count:
#     if word not in source_word2idx:
#       source_word2idx[word] = len(source_word2idx)

#   for word, _ in target_count:
#     if word not in target_word2idx:
#       target_word2idx[word] = len(target_word2idx)

#   source_idx2word = {v:k for k, v in source_word2idx.items()}
#   source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()
#   for sentence in root:
#     text =  sentence.find('text').text.lower()
#     #text = clean_text(sentence.find('text').text.lower())
#     #text = ' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', text).lower().split())
#     text = re.sub(r'[.,:;/\-?!\"\n()\\]',' ', text).lower()
#     #print text
#     if len(text.strip()) != 0:
#       for asp_terms in sentence.iter('aspectTerms'):
#         for asp_term in asp_terms.findall('aspectTerm'):
#           pos_info, lab = _get_data_tuple(text, asp_term.get('term').lower(), int(asp_term.get('from')), int(asp_term.get('to')), asp_term.get('polarity'), source_word2idx)
#           boobs[lab] += 1
#           if lab == 3:
#             continue

#           idx = []
#           #for word in word_tokenize(text):
#           for word in text.split():
#             idx.append(source_word2idx[word])
          
#           refined_idx = []
#           refined_pos_info = []
#           aspect_count = 0
#           for index in range(len(pos_info)):
#             #if pos_info[index] <=100 and pos_info[index] != 0 and source_idx2word[idx[index]] not in stop:
#             if pos_info[index] <=100 and pos_info[index] != 0:
#               refined_idx.append(idx[index])
#               refined_pos_info.append(pos_info[index])
#             # elif pos_info[index] == 0 and aspect_count == 0:
#             #   refined_idx.append(source_word2idx['$T$'])
#             #   refined_pos_info.append(pos_info[index])
#             #   aspect_count += 1

            
#           if len(refined_idx)==0:
#             print "Error", refined_idx, [source_idx2word[id] for id in refined_idx], pos_info
#           source_data.append(refined_idx)
#           source_loc_data.append(refined_pos_info)
#           target_data.append(target_word2idx[asp_term.get('term').lower()])
#           target_label.append(lab)

#   print("Read %s aspects from %s" % (len(source_data), fname))
#   print boobs
#   return source_data, source_loc_data, target_data, target_label, max_sent_len