import os
from core.utils import bang_nguyen_am
import random
vietnamese_vowels = set()

for group in bang_nguyen_am:
    vietnamese_vowels.update(group[:-1])
vietnamese_vowels = list(vietnamese_vowels)

def change_word(word:str,p,min_chars = 3):
    '''
    p: amount of characters to be chaged
    '''
    assert p >=0 and p <= 1
    if len(word) < min_chars:
        return word
    num_char = int(len(word)*p)
    if p> 0 and num_char == 0  :
         num_char = 1

    # Convert the word into a list to allow mutation
    word_list = list(word)

    # Randomly select `p` unique indices to change
    indices_to_change = random.sample(range(len(word_list)), num_char)

    # Replace the selected indices with random lowercase characters
    for idx in indices_to_change:
        word_list[idx] = random.choice(vietnamese_vowels)  # Random lowercase letter

    # Join the list back into a string and return
    return ''.join(word_list)
def reduce_word(word:str,num_take,drop = 'last',min_chars = 4):
    '''
     drop: last or middle'''
    assert num_take >= 1
    assert drop in ['last', 'middle'] 
    
    if num_take >= len(word) and len(word) < min_chars:
        return word
    
    if drop == 'last':
        return word[:num_take]
    
    else:
        end_index = -num_take // 2
        begin = num_take + end_index
        return word[:begin] + word[end_index:]
import random

class Augment:
    def __init__(self, augment_args):
        '''
        Initialize the augmentation class.

        Parameters:
        augment_args (dict): A dictionary of function names and their parameters.
        '''
        self.augment_args = augment_args
        self.compose = []
        self.percentage_of_augment = self.augment_args.get('percentage_of_augment', 0.2)

        # Collect valid augmentation functions and their arguments
        for fn_name, fn_args in self.augment_args.items():
            fn = getattr(self, fn_name, None)
            if callable(fn):
                self.compose.append((fn, fn_args))

    def __call__(self, sentence: str) -> str:
        '''
        Apply random augmentation functions to some words in the sentence.

        Parameters:
        sentence (str): The input sentence to augment.

        Returns:
        str: The augmented sentence.
        '''
        words = sentence.split()
        num_words_to_augment = max(1, int(len(words) * self.percentage_of_augment))
        words_to_augment = random.sample(words, num_words_to_augment)

        # Apply random augmentation functions to selected words
        for i, word in enumerate(words):
            if word in words_to_augment:
                fn, args = random.choice(self.compose)  # Randomly select a function
                words[i] = fn(word, **args)

        return ' '.join(words)

    # Example augmentation functions
    def change_word(self, word: str, p: float) -> str:
        '''
        Modify `p` fraction of characters in the input `word`.

        Parameters:
        word (str): The input word to modify.
        p (float): Fraction (0 ≤ p ≤ 1) of characters to be changed.

        Returns:
        str: The modified word.
        '''
        num_char = int(len(word) * p)
        if p > 0 and num_char == 0:
            num_char = 1

        word_list = list(word)
        indices_to_change = random.sample(range(len(word_list)), num_char)
        for idx in indices_to_change:
            word_list[idx] = random.choice(vietnamese_vowels)

        return ''.join(word_list)

    def reduce_word(self, word: str, num_take: int, drop: str = 'last') -> str:
        '''
        Reduce the input word to a specified number of characters.

        Parameters:
        word (str): The input word to reduce.
        num_take (int): Number of characters to keep.
        drop (str): Reduction mode: 'last' (drop from end) or 'middle' (drop from middle).

        Returns:
        str: The reduced word.
        '''
        if num_take >= len(word):
            return word

        if drop == 'last':
            return word[:num_take]

        # drop == 'middle'
        begin = num_take // 2
        end_index = -(num_take - begin)
        return word[:begin] + word[end_index:]

    # Example augmentation functions
    def change_word(self, word: str, p: float) -> str:
        return change_word(word, p)
    
    def reduce_word(self, word: str, num_take: int, drop: str = 'last') -> str:
        return reduce_word(word, num_take, drop)
          
def get_augument(augument_config = None):
    # Test data
    sentence = "sao mn mỉa mai ác thế"
    augment_args = {
        'percentage_of_augment': 0.5,  # 50% of the words will be augmented
        'change_word': {'p': 0.5},
        'reduce_word': {'num_take': 3, 'drop': 'middle'}
    }

    # Initialize the Augment class
    augment = Augment(augment_args)
    return augment


if __name__ == "__main__":
    from core.utils import tien_xu_li
    augment = get_augument()
    file_path = r"C:\Users\9999\phuoc\paper\ViLexNorm\EnhancingViLexNorm\data\train.csv"
    import pandas as pd
    csv_data = pd.read_csv(file_path)
    csv_data['normalized'] = csv_data.normalized.apply(tien_xu_li)
    csv_data['original'] = csv_data.original.apply(tien_xu_li)
    

    csv_data['original'] = csv_data.original.apply(augment.__call__)
    
    csv_data = csv_data[['original','normalized']]
    csv_data.to_csv('out_augmented.csv')  

    

# def test_augment():
#     # Test data
#     sentence = "sao mn mỉa mai ác thế"
#     augment_args = {
#         'percentage_of_augment': 0.5,  # 50% of the words will be augmented
#         'change_word': {'p': 0.5},
#         'reduce_word': {'num_take': 3, 'drop': 'middle'}
#     }

#     # Initialize the Augment class
#     augment = Augment(augment_args)

#     # Test the augmentation process
#     augmented_sentence = augment(sentence)

#     print(augmented_sentence)
# # Run the test
# test_augment()
