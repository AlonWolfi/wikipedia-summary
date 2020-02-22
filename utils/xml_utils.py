import xml
import shutil
import xml.etree.ElementTree as ET
import re


def remove_first_line_nonesense_xml(path):
    '''
    parse and change xml file in order to get only the opening and ending of the first line of the file.
    '''

    # Read file
    with open(path, 'r', encoding="utf8") as xml_file:
        # Get first line and reminder
        first_line = xml_file.readline()
        reminder = ''.join(xml_file.readlines())

    # Edit first line
    xml_parts = first_line.split(' ')
    xml_opening = xml_parts[:1]
    xml_ending = xml_parts[-2:]
    first_line_edited = ' '.join(xml_opening + xml_ending)

    # Write to file
    with open(path, 'w', encoding="utf8") as xml_file:
        xml_file.write(first_line_edited)
        xml_file.write(reminder)


def get_root_of_file(xml_path):
    '''
    @return the root xml element of a file
    '''
    tree = ET.parse(xml_path)
    return tree.getroot()


def generate_pages(xml_path):
    '''
    @return: a generator for the text of each page of the xml file
    '''
    root = get_root_of_file(xml_path)
    pages = root.findall('page')
    for page in pages:
        try:
            yield page.find('title').text, page.find('revision').find('text').text
        except:
            yield None, page.find('revision').find('text').text
