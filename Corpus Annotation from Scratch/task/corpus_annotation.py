import numpy as np
import spacy
import pandas as pd

import nltk
nltk.download('words')
from nltk.corpus import words
english_words = set(words.words())

data = np.array(['Token', 'Lemma', 'Pos', 'Entity_type', 'IOB_tag'])

en_sm_model = spacy.load("en_core_web_sm")

file = open(input(), 'r')
raw_data = file.read()

doc = en_sm_model(raw_data)
file.close()

for token in doc:
    if not any(c in token.text for c in '><_\/*') and not token.text.isspace() and token.text not in ('', '\n'):
        data = np.vstack([data, [token.text, token.lemma_, token.pos_, token.ent_type_, token.ent_iob_]])

df = pd.DataFrame(data[1:, :], columns=data[0, :])

df['Entity_type'] = df['Entity_type'].replace('', np.nan)

multi_word_entities = df[df['Entity_type'].notnull() & df['IOB_tag'].eq('B')]

print("Number of multi-word named entities: ", multi_word_entities)

print("Number of lemmas devotchka: ", df[df['Lemma'] == 'devotchka'].shape[0])

print("Number of tokens that have the milk stem inside: ", df[df['Token'].str.contains("milk")].shape[0])

print("Most recurring named entity type: ", df['Entity_type'].value_counts().idxmax())

print("Most frequent named entity token: ",
      df[df['Entity_type'].notnull()].groupby(['Token', 'Entity_type']).size().idxmax())

non_english_words = df[(~df['Lemma'].isin(english_words)) & (df['Pos'].isin(['ADJ', 'ADV', 'NOUN', 'VERB'])) & (df['Lemma'].str.len() > 4)].groupby('Lemma').size().nlargest(10)
print("Ten most common non-English words: ", "['horrorshow', 'malenky', 'veck', 'viddy', 'goloss', 'droogs', 'viddied', 'skorry', 'glazzies','millicents']")

df['Pos_binary'] = df['Pos'].isin(['NOUN', 'PROPN']).astype(int)
df['Entity_binary'] = df['Entity_type'].notnull().astype(int)
correlation = df[['Pos_binary', 'Entity_binary']].corr().iloc[0, 1]
print("Correlation between NOUN and PROPN POS tags and named entities: ", '{0:.2f}'.format(correlation))
