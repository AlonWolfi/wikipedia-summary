import wikipedia
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
import wptools

from pathlib import Path

PROJECT_NAME = 'wikipedia-summary'

def get_project_dir() -> Path:
    '''
    @return: The path of the project
    '''
    current_path = Path.cwd()
    while current_path.name != PROJECT_NAME:
        current_path = current_path.parent
    return current_path

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

DATA_DIR = get_project_dir() / 'cache/page_list.txt'

with open(DATA_DIR, 'r', encoding="utf8") as f:
    data = f.read()
pages_lst = data.split('\n')

docs = []
i=0
j=0
wikipedia_docs = []
wiki_wiki_docs = []

pages_lst = pages_lst[:3]

for p in pages_lst:
    i += 1
    print(i)
    try:
        wikipedia_docs += [wikipedia.page(p).content]
        print('wikipedia ' + str(i))
    except Exception as e:
        print(e)
    try:
        wiki_wiki_docs += [load_text(p)]
        print('wikiwiki ' + str(i))
    except Exception as e:
        print(e)

for doc1, doc2 in zip(wikipedia_docs, wiki_wiki_docs):
    print('#'*20)
    print(doc1)
    print(doc2)