from dataclasses import dataclass, field
import pandas as pd

import hand_tracker as ht

@dataclass
class Pred_letter:
    letter: str
    mapped_pos: list[int] = field(default_factory=list)
    is_mapped: bool = False

similar_letter = {
            'A': ['S'],
            'B': ['E'],
            'C': ['O'],
            'D': ['O', 'C', 'L'],
            'E': ['B'],
            'F': [],
            'G': ['H'],
            'H': ['G'],
            'I': ['Y'],
            'J': ['I', 'Y'],
            'K': [],
            'L': ['D'],
            'M': ['N', 'A'],
            'N': ['M', 'A'],
            'O': ['D', 'C', 'P'],
            'P': ['Q', 'M', 'G'],
            'Q': ['N', 'P'],
            'R': ['U'],
            'S': ['A'],
            'T': [],
            'U': ['R', 'K', 'B'],
            'V': ['U', 'R', 'K'],
            'W': [],
            'X': ['G'],
            'Y': ['I'],
            'Z': ['L', 'P', 'X', 'M'],
        }

def parse_feedback_file(txt_file_name, true_words):
    # Put letters into respective objects
    file = open(txt_file_name)
    file_words = file.readlines()
    assert(len(file_words) == len(true_words))
    # print(len(file_words))
    # print(len(true_words))

    for file_word, true_word in zip(file_words, true_words):
        pred, true = seperate_letters(file_word, true_word)
        success, pred, true = map_true_to_pred(pred, true)
        if not success:
            print("Too much uncertainty!")
        print_pred_letter_list(true)
        print_pred_letter_list(pred)
        edit_csv("hund.csv", pred, true)


def seperate_letters(file_word, true_word):
    pred_letters = []
    true_letters = []
    for l in file_word[:-1]:
        pred_letters.append(Pred_letter(letter = l))
    for l in true_word:
        true_letters.append(Pred_letter(letter = l))

    return pred_letters, true_letters

def map_true_to_pred(pred_letters, true_letters):
    # Map true letters to predicted ones
    # First pass: Exact Match
    map_min = -1
    for i_t, l_t in enumerate(true_letters):
        for i_p, l_p in enumerate(pred_letters):
            if l_t.is_mapped and not l_p.is_mapped and l_p.letter == l_t.letter:
                l_t.mapped_pos.append(i_p)
                l_p.mapped_pos.append(i_t)
                l_p.is_mapped = True
                map_min = i_p
            elif l_t.is_mapped:
                break
            if not l_p.is_mapped and not l_t.is_mapped and l_t.letter == l_p.letter and i_p > map_min:
                l_t.mapped_pos.append(i_p)
                l_t.is_mapped = True
                l_p.mapped_pos.append(i_t)
                l_p.is_mapped = True
                map_min = i_p

    if count_unmapped(true_letters) > 0:
        # Second pass: Similar Letters
        for i_t, l_t in enumerate(true_letters):
            if not l_t.is_mapped:
                for i_p, l_p in enumerate(pred_letters):
                    if l_p.is_mapped and len([*filter(lambda x: x >= i_t, l_p.mapped_pos)]) > 0:
                        # We are outside of our little slice
                        break
                    elif not l_p.is_mapped and l_p.letter in similar_letter[l_t.letter]:
                        l_t.mapped_pos.append(i_p)
                        l_t.is_mapped = True
                        l_p.mapped_pos.append(i_t)
                        l_p.is_mapped = True

    if count_unmapped(pred_letters) > 0:
        for i_p, l_p in enumerate(pred_letters):
            # Look at previous
            if not l_p.is_mapped and i_p > 0 and pred_letters[i_p - 1].is_mapped:
                prev = pred_letters[i_p - 1]
                mapped_true_index = prev.mapped_pos[len(prev.mapped_pos)-1]
                prev_mapped_true = true_letters[mapped_true_index]
                if l_p.letter in similar_letter[prev_mapped_true.letter]:
                    prev_mapped_true.mapped_pos.append(i_p)
                    l_p.mapped_pos.append(mapped_true_index)
                    l_p.is_mapped = True
            # Look at next
            if not l_p.is_mapped and i_p < (len(pred_letters) - 1) and pred_letters[i_p + 1].is_mapped:
                next = pred_letters[i_p + 1]
                mapped_true_index = next.mapped_pos[0]
                next_mapped_true = true_letters[mapped_true_index]
                if l_p.letter in similar_letter[next_mapped_true.letter]:
                    next_mapped_true.mapped_pos.append(i_p)
                    l_p.mapped_pos.append(mapped_true_index)
                    l_p.is_mapped = True
    
    success = True if (count_unmapped(true_letters) < (len(true_letters)/2)) else False
    return success, pred_letters, true_letters

def count_unmapped(list_of_letters):
    count = 0
    for letter in list_of_letters:
        if not letter.is_mapped:
            count = count + 1
    return count

def print_pred_letter_list(list_of_letters):
    # print("print func")
    for letter in list_of_letters:
        print("{} {}".format(letter.letter, letter.mapped_pos))
    print("")

def edit_csv(csv_file_name, pred_letters, true_letters):
    data_frame = pd.read_csv(csv_file_name)
    classes = data_frame["class"]
    pred_i = 0
    letter_count = 0
    counting_letter = ''
    counted_list = []

    for index, letter in enumerate(classes):
        # Skip unmapped letters
        while not pred_letters[pred_i].is_mapped:
            pred_i = pred_i + 1
        if letter != pred_letters[pred_i].letter and counting_letter != '':
            pred_i = pred_i + 1
            counting_letter = ''
        if letter != pred_letters[pred_i].letter:
            letter_count = 0
            data_frame.at[index, 'class'] = 'junk1'
        elif letter == pred_letters[pred_i].letter and letter_count == ht.FRAMES_TO_PRINT:
            counting_letter = letter
            print(counting_letter)
            #Predicted letter is not mapped
            if not pred_letters[pred_i].is_mapped:
                for i in counted_list:
                    data_frame.at[i, 'class'] = 'junk2'
                data_frame.at[index, 'class'] = 'junk3'
            elif pred_letters[pred_i].letter != true_letters[pred_letters[pred_i].mapped_pos[0]].letter:
                # Predicted letter is mapped to a different true letter
                data_frame.at[index,'class'] = true_letters[pred_letters[pred_i].mapped_pos[0]].letter
                for i in counted_list:
                    data_frame.at[i,'class'] = true_letters[pred_letters[pred_i].mapped_pos[0]].letter
            letter_count = letter_count + 1
        elif letter == pred_letters[pred_i].letter:
            letter_count = letter_count + 1
            counted_list.append(index)
    
    data_frame['class'] = classes
    data_frame.to_csv("hund2.csv", header="True", index=False)


    

my_list = ["HUND"]
parse_feedback_file("predict.txt", my_list)
