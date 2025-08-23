def remove_specific_characters(text):
    for char in [';', ',', '-', '.', "'", '*', ' ', ":", '_', '+', '"']:
        text = text.replace(char, '')
    return text

def upperCase(text):
    return text.upper()

def is_letter(char):
    return char.isalpha()

def is_number(char):
    return char.isdigit()

def build_char_map(mapping_groups):
    """Convert multi-key mapping groups into a flat character map."""
    char_map = {}
    for keys, value in mapping_groups.items():
        for key in keys:
            char_map[key] = value
    return char_map

def map_character(char, char_map):
    return char_map.get(char, char)

def correct_text(text, mapping_groups):
    char_map = build_char_map(mapping_groups)
    return ''.join(map_character(ch, char_map) for ch in text)

def cleanUpPlate(plate):
    if plate == '':
        return None
    cleaned_plate = remove_specific_characters(plate)
    uppercased_cleaned_plate = upperCase(cleaned_plate)
    new_uppercased_cleaned_plate = ''
    if len(uppercased_cleaned_plate) > 9:
        for character in uppercased_cleaned_plate:
            if is_number(character) or is_letter(character):
                new_uppercased_cleaned_plate += character
    else:
        new_uppercased_cleaned_plate = uppercased_cleaned_plate
    if len(new_uppercased_cleaned_plate) <4:
        return None
    third = new_uppercased_cleaned_plate[2]
    if not is_letter(third):
        third = correct_text(third, char_map)

    other = new_uppercased_cleaned_plate[:2] + new_uppercased_cleaned_plate[3:]
    new_other = ''

    for character in other:
        if not is_number(character):
            character = correct_text(character, char_map_for_number)
        new_other += character

    return new_other[:2]+ '-' + third + new_other[2] + ' ' + new_other[3:]

# MAP DICTIONARY
char_map = {
    ('5', ): 'S',
    ('['): "L",
    ('1'): "Y",
    ('0'): "D",
    ('8'): "B",
    ('4'): "A",
    ('2'): "Z",
    ('6'): 'G',
    ('[', ']'): 'L',
    ('7'): 'V',

}

char_map_for_number = {
    ("L", 'A'): '4',
    ("I", "[", "]", "N", "H", "T", 'Y'): '1',
    ("Q"): '0',
    ('Z'): '2',
    ('B', 'U'): '8',
    ('P'): '3',
    ('J', 'S'): '5',
    ('O', 'D'): '0',
    ('G'): '6',
    ('V'): '7'
}

