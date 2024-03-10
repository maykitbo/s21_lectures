import spacy
# from tqdm.notebook import tqdm
import re
import progressbar


def progress_bar(data):
    bar = progressbar.ProgressBar(maxval=len(data)).start()
    for i, element in enumerate(data):
        yield element
        bar.update(i)



# Load English and Russian models
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")


def limitisation(data):
    return [_linit_text(text) for text in progress_bar(data)]


def _linit_text(text_tuple):
    return (' '.join(token.lemma_ for token in nlp_ru(text_tuple[0])),
            ' '.join(token.lemma_ for token in nlp_en(text_tuple[1])))


def normalization(data):
    return [_norm_text(text) for text in progress_bar(data)]


def _norm_text(text_tuple):
    return (' '.join(token.text.lower() for token in nlp_ru(text_tuple[0])),
            ' '.join(token.text.lower() for token in nlp_en(text_tuple[1])))


def normolize_ru_sentence(sentence):
    return ' '.join(token.text.lower() for token in nlp_ru(sentence))

