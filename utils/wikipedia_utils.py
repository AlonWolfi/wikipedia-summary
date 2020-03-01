import os
import glob
import nltk
import gensim
import pandas as pd
import wikipedia
import wikipediaapi
import wptools

wiki_wiki = wikipediaapi.Wikipedia('en')

def load_text(page_name):
    f = wiki_wiki.page(page_name)
    if f.exists():
        return f.text
    else:
        print('Page not found: ' + page_name)
        return None

def load_info_box(page_name):
    so = wptools.page(page_name).get_parse()
    infobox = so.data['infobox']
    return infobox