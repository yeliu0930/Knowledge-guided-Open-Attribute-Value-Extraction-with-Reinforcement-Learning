import unicodedata
import pandas as pd
import similarity_metrics
from functools import reduce



def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False
  
def token_word(answer_value):
  temp_word = ""
  for c in answer_value:
    if not _is_whitespace(c):
      temp_word += c
  temp_word = temp_word.lower()
  return temp_word


def get_sim_features(ans, reference_values, do_bert = True):
  #ans_emb = word_vector(ans)
  cols = ['f_levenm', 'f_lcsm', 'f_levenr', 'f_lcsr', 'f_leven', 'f_lcs', "d_len"]
  sims = pd.DataFrame(columns = cols)
  for value in reference_values: 
    if do_bert:
      sims.loc[len(sims)] = list(similarity_metrics.GetFeature(token_word(str(value)), ans)[:7])  
    else:
      sims.loc[len(sims)] = list(similarity_metrics.GetFeature(str(value),ans)[:7])  
  #l = sims.max().tolist(), sims.min().tolist(), sims.mean().tolist(),\
  #    sims.median().tolist()
  l = sims.max().tolist(), sims.mean().tolist()
  result = reduce(lambda x,y: x+y,l)
  return result
  