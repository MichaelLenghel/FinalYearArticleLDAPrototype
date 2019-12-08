import os
import re
from pprint import pprint

# NLP libraries
import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel, KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import spacy
import pattern
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# Include extra words. May in the future extend it even further
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# sklearn for accessing 20 news groups
from sklearn.datasets import load_files

def gen_bunch(news_path):
    # Cateogies in data set
    news_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc'
                , 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'
                , 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball'
                , 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med'
                , 'sci.space', 'soc.religion.christian', 'talk.politics.guns'
                , 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    # Setup path to test corpus
    NEWS_GROUPS_TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), news_path))

    # Print the path
    # print(NEWS_GROUPS_TEST_PATH)


    ##### Need to implement a method for including custom categories! ####

    # Load all test data.

    # news_test = load_files(NEWS_GROUPS_TEST_PATH, description='News Paper Test Topics from 20 news groups'
    #                             , categories=news_categories, load_content=True , shuffle=False, encoding='latin1'
    #                             , decode_error='strict')

    # Shuffling the data in order to increase distribution of topics and not overly simplify NLP patterns
    # news_test.data is everything in one big string
    news_test = load_files(NEWS_GROUPS_TEST_PATH, description='News Paper Test Topics from 20 news groups'
                                , categories=news_categories, load_content=True , shuffle=True, encoding='latin1'
                                , decode_error='strict', random_state=30)
    
    # Note:
    # Shows the topic and document ID + the article.
    # print(news_test.filenames[0])
    # print(news_test.data[0])

    # Get all of the file names
    # for integer_category in news_test.target[:10]:
    #     print(news_test.target_names[integer_category])

    return news_test

def multiple_replacements(article):
    empty_str = ""

    # Replacing all dashes, equals, cursors
    replacements = {
        "-" : empty_str,
        "=": empty_str,
        "^": empty_str,
    }

    # Replace newlines
    article_list = re.sub('\s+', ' ', article)

    # Replace emails
    article_list = re.sub('\S*@\S*\s?', '', article)

    # Replace quotes
    article_list = re.sub("\'", "", article)

    # Create a regular expression using replacements and join them togetehr.
    # re.compile creates the pattern object
    # re.escape avoids using special characters in regex
    reg = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))

    # For each match, look-up corresponding value in dictionary
    return reg.sub(lambda value: replacements[value.string[value.start():value.end()]], article)


def split_to_word(articles):
    # Iterate over every article 
    for article in articles:
        # Yield to not overload the memory with the big data set.
        # Deacc parameter removes all punctuations as well as spliting each word.
        yield(gensim.utils.simple_preprocess(str(article), deacc=True))

def create_bigrams(articles, bigram_model):
    return [bigram_model[article] for article in articles]

def remove_stopwords(articles):
    return [[w for w in simple_preprocess(str(article)) if w not in stop_words] for article in articles]

def lemmatize_words(bigram_model):
    # Only considers nouns, verbs, adjectives and adverbs
    return [[w for w in lemmatize(str(article))] for article in bigram_model]

# This method is about a minute faster for a data set of 7000 than the one above
def lemmatization(articles, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    articles_lem = []

    # Load the spacy lammatixation model for english 
    spacy_lem = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    for article in articles:
        w = spacy_lem(" ".join(article)) 
        articles_lem.append([token.lemma_ for token in w if token.pos_ in allowed_postags])

    return articles_lem

def build_lda_model(articles, word_dict, corpus):
    # Build LDA model
    # Retry with random_state = 0
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=word_dict,
                                            num_topics=50,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
    
    return lda_model

def clean_data(test_bunch):
    ########## ALGO ###################

    # Create a list of the articles
    article_list = list(test_bunch.data)

    # Replace =, cursor and dash, quotes, newlines and emails
    article_list = [multiple_replacements(article) for article in article_list]

    # split each article into words
    article_word_list = list(split_to_word(article_list))

    # Print the first article word by word:
    # print(article_word_list[0])

    # brigrams model
    # Need bigrams as it cuts word that go together down into one
    bigram = gensim.models.Phrases(article_word_list, min_count=8, threshold=100)

    bigram_model = gensim.models.phrases.Phraser(bigram)

    # Print bigrams
    # print(bigram_model[article_word_list[0]])

    # Remove stopwords
    article_no_stopwords = remove_stopwords(article_word_list)

    # make bigrams
    bigram_words = create_bigrams(article_no_stopwords, bigram_model)

    # Lemmatize - By default only nouns, verbs, adjectives and adverbs
    # lemmatized_article = lemmatize_words(bigram_words)

    lemmatized_article = lemmatization(bigram_words, allowed_postags=['NOUN', 'VERB', 'ADJ',  'ADV'])

    return lemmatized_article

def save_model(lda_model):
    # Below doesn't work due to access denied issues, datapath is the alternative

    # MODEL_PATH = "../models/"
    # model_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))

    model_file = datapath("model")
    lda_model.save(model_file)

def save_data(newspaper_list):
    MODEL_PATH = "../newspaper_data/newspaper_list.txt"
    newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))
    
    with open(newspaper_list_file, 'w') as filehandle:
        for listitem in newspaper_list:
            filehandle.write('%s\n' % listitem)


def main():
    TEST_CORPUS_PATH = '../corpus/20news-bydate-test/'

    # Create bunch data structure with all data from articles
    test_bunch = gen_bunch(TEST_CORPUS_PATH)
    print('Finished loading newspaper data...')

    # Clean data (Remove stopwords, characters, lementize, etc.)
    # Format is list of articles, each article containg another list of words
    test_data_cleaned = clean_data(test_bunch)
    print('Finished cleaning data...')

    # save_newspaper_data
    save_data(test_data_cleaned)
    print('Finished saving cleaned_data...')
    

    # print(test_data_cleaned[0])
    # print(test_data_cleaned[:1])

    # Create dictionary. This maps id to the word
    word_dict = corpora.Dictionary(test_data_cleaned)

    # Create corpus. This directly contains ids of the word and the frequency.
    corpus = [word_dict.doc2bow(data) for data in test_data_cleaned]
    print('Finished creating corpus...')

    # Will map word id to word frequency
    #  print(corpus[:1])

    # Print wname of word and freqnecy
    # for c in corpus[:1]:
    #     for word_id, frequency in c:
    #         print(word_dict[word_id], " = ", frequency)

    # Create the lda model. When using the method below ut returns None
    lda_model = build_lda_model(test_data_cleaned, word_dict, corpus)
    print('Finished building lda model...')

    # Print out contents of lda_model
    # pprint(lda_model.print_topics())

    # Save the model
    save_model(lda_model)
    print('Finished saving lda model...')


if __name__ == '__main__':
    main()
