import os
from pprint import pprint

# NLP libraries
import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess, lemmatize
from gensim.models import CoherenceModel, KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords

# Visualisation libraries.
import pyLDAvis.gensim

# Get the model from genism path
def retrieve_modal():
    # Path to model
    model_file = datapath("model")
    # Load
    lda = gensim.models.ldamodel.LdaModel.load(model_file)

    print('Finished retrieving model...')
    return lda

def retrieve_word_article_list():
    MODEL_PATH = "../newspaper_data/newspaper_list.txt"
    newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))

    MODEL_PATH_WRITE = "../newspaper_data/newspaper_list_writing.txt"
    newspaper_list_file_write = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH_WRITE))

    newspaper_article_list = []
    newspaper_word_list = []

    with open(newspaper_list_file, 'r') as filehandle:
        for line in filehandle:
            # String splitting removes first bracket and new line + closing bracket
            current_line = line[1:-2]

            # Split each word into the list
            current_list = current_line.split(', ')

            for word in current_list:
                # Append the word and remove closing and opening quotation
                newspaper_word_list.append(word[1:-1])
                
            newspaper_article_list.append(newspaper_word_list)
            newspaper_word_list = []

    print('Finished retrieving article word list...')
    return newspaper_article_list

def build_dictionary_corpus(article_word_list):
    # Make word_dict & corpus
    word_dict = corpora.Dictionary(article_word_list)
    corpus = [word_dict.doc2bow(data) for data in article_word_list]

    print('Finished creating word_dict and corpus...')
    return word_dict, corpus

def build_lda_model(articles, word_dict, corpus):
    # Build LDA model
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

def compute_complexity(article_word_list, word_dict, corpus, lda):

    
    # Coherence Score: 0.4132613386862506
    # Coherence score is the probability that a word has occured in a certain topic, so the quality of the topic matching.
    coherence_model_lda = CoherenceModel(model=lda, texts=article_word_list, dictionary=word_dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score:', coherence_lda)

    # Perplexity: -21.294153422496972
    # Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus.
    # Also output the calculated statistics, including the perplexity=2^(-bound), to log at INFO level.
    print('Perplexity:', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

def build_lda_visualizaiton(corpus, word_dict, lda):
    MODEL_PATH = "../topic_visualisaiton/visualisation.html"
    topic_visualisation = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))

    # Extracts info from lda model for visualisaition
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, word_dict, sort_topics=False)
    print('Finished Prepare!!')

    pyLDAvis.show(lda_display)
    ('Finished display!')
    # Save graph so it can be used. IPYTHON dependancy required to display
    pyLDAvis.save_html(lda_display, topic_visualisation)
    print('Finished saving visualization...')


def main():
    # Recover list data
    article_word_list = retrieve_word_article_list()
    
    # Create dictionary, corpus
    word_dict, corpus = build_dictionary_corpus(article_word_list)

    # Retrieve ldamodel
    lda = retrieve_modal() 

    ####### Compute complexity (Left out as long execution) ########
    compute_complexity(article_word_list, word_dict, corpus, lda)

    # Below method only works in a shell such as python shell. Also takes about 15 minutes
    # build_lda_visualizaiton(corpus, word_dict, lda)

    # get_document_topics with a single token 'religion'
    # text = ["religion"]
    # bow = word_dict.doc2bow(text)
    # pprint(lda.get_document_topics(bow))

    # # View topics in LDA model (Print keywords)
    # Compute scores
    # Visualise data on graphs

    # # Show top 30 topics (with weights)
    # pprint(lda.show_topics(30))

    # # Display topics and simply list words without weights for more clarity
    # for index, topic in lda.show_topics(formatted=False, num_words= 30):
    #     print('Topic: {} \nWords: {}'.format(index, '|'.join([w[0] for w in topic])))


if __name__ == '__main__':
    main()

