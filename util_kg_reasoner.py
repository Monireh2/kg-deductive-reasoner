from __future__ import division

import sys
import time

import numpy as np

from memn2n.memory import MemoryL, MemoryBoW
from memn2n.nn import AddTable, CrossEntropyLoss, Duplicate, ElemMult, LinearNB, Linear
from memn2n.nn import Identity, ReLU, Sequential, LookupTable, Sum, Parallel, Softmax

# for saving model after training
import gzip
import pickle
from datetime import datetime


def load_all_data(data_files, dictionary):
    """
    This function load the whole data into memory

    :return:
        lines_data, dictionary


    """
    lines_data = []
    for fp in data_files:
        with open(fp) as f:
            lines = f.readlines()
            lines_data.extend(lines)

    for line_idx, line in enumerate(lines_data):
        line = line.rstrip().lower().replace('\'', '')
        words = line.split()

        # Skip substory index
        for k in range(1, len(words)):
            w = words[k]
            if w.endswith('.') or w.endswith('?'):
                w = w[:-1]
            if w not in dictionary:
                dictionary[w] = len(dictionary)

    return lines_data, dictionary


def generate_next_story_helper(lines_data, dictionary):
    """
    This returns the data of a story

    :returns: story, questions, qstory, new_start


    """

    # size limitation settings
    restrict_bl = True
    limit_word = 4
    limit_sentence = 1000
    limit_question = 1000
    # changed for Bosch run phython reasoner
    #limit_question = 20000


    # Number of words in each sentence, number of sentences in each story, number of stories
    story = np.zeros((limit_word, limit_sentence, 1), np.int16)

    # Question's meta data, number of questions
    questions = np.zeros((14, limit_question), np.int16)

    # number of words in each question, number of questions
    qstory = np.zeros((limit_word, limit_question), np.int16)


    story_idx, question_idx, sentence_idx, max_words, max_sentences = -1, -1, -1, 0, 0

    for line_idx, line in enumerate(lines_data):

        line = (line.rstrip().lower()).replace('\'', '')
        words = line.split()
        # to limit memory usage
        if restrict_bl:
            
            if len(words) > limit_word:
                if words[-1] == 'yes' or words[-1] == 'no':
                    words = words[:limit_word]+[words[-1]]

                else:
                    words = words[:limit_word]


            if sentence_idx >= limit_sentence-1:
                if words[-1] != 'yes' and words[-1] != 'no' and words[0] != '1':
                    continue

            if question_idx >= limit_question//2-1:
               if words[-1] == 'yes':
                   continue

            if question_idx >= limit_question-1:
               if words[-1] == 'no':
                   continue


        # *** return when it comes to the next story ***
        if words[0] == '1' and story_idx != -1:

            #story = story[:max_words, :max_sentences, :(story_idx + 1)]
            #questions = questions[:, :(question_idx + 1)]
            # random shuffling questions to not put yes at the end
            #questions=questions[:, np.random.permutation(questions.shape[1])]
            #qstory = qstory[:max_words, :(question_idx + 1)]

            new_start = line_idx

            return story, questions, qstory, new_start

        # Story begins
        if words[0] == '1' and story_idx == -1:
            story_idx += 1
            sentence_idx = -1


        # FIXME: This condition makes the code more fragile!
        if words[-1] != 'yes' and words[-1] != 'no':
            is_question = False
            sentence_idx += 1
            # To skip the number
            #words = words[1:]
        else:
            is_question = True
            question_idx += 1
            questions[0, question_idx] = story_idx
            questions[1, question_idx] = sentence_idx
            # To skip the number
            #words = words[1:]

        # Skip substory index
        for k in range(1, len(words)):
            w = words[k]

            if w.endswith('.') or w.endswith('?'):
                w = w[:-1]

            if w not in dictionary:
                dictionary[w] = len(dictionary)

            if max_words < k:
                max_words = k

            if not is_question:
                story[k - 1, sentence_idx, story_idx] = dictionary[w]
            # is_question
            else:
                # index of each word of question
                qstory[k - 1, question_idx] = dictionary[w]

                # NOTE: Punctuation is already removed from w
                if k + 1 == len(words) - 1:
                    if words[k + 1] == 'yes' or words[k + 1] == 'no':
                        answer = words[k + 1]
                        if answer not in dictionary:
                            dictionary[answer] = len(dictionary)

                        questions[2, question_idx] = dictionary[answer]
                        questions[-1, question_idx] = line_idx
                        break

        if max_sentences < sentence_idx + 1:
            max_sentences = sentence_idx + 1



    #story = story[:max_words, :max_sentences, :(story_idx + 1)]
    #questions = questions[:, :(question_idx + 1)]
    # random shuffling questions to not put yes at the end
    #questions = questions[:, np.random.permutation(questions.shape[1])]
    #qstory = qstory[:max_words, :(question_idx + 1)]

    # all data have been processed, set start = -1 as an end flag
    new_start = -1
    return story, questions, qstory, new_start



def generate_next_story(lines_data, dictionary):
    """
    This generator yields the next story

    :return:
        story, questions, qstory, start

  
    """
    while True:
        story, questions, qstory, start = generate_next_story_helper(lines_data, dictionary)
        print("\n*** New Story, Current time: {} ***\n".format(datetime.now()))
        yield story, questions, qstory, np.array(start)


def parse_kg_task(data_files, dictionary, include_question):
    """ Parse Kg data.

    Args:
       data_files (list): a list of data file's paths.
       dictionary (dict): word's dictionary
       include_question (bool): whether count question toward input sentence.

    Returns:
        A tuple of (story, questions, qstory):
            story (3-D array)
                [position of word in sentence, sentence index, story index] = index of word in dictionary
            questions (2-D array)
             ---- To represent the question's meta data ----
                [0-9, question index], in which the first component is encoded as follows:
                    0 - story index
                    1 - index of the last sentence before the question
                    2 - index of the answer word in dictionary
                    3 to 13 - indices of supporting sentence
                    14 - line index
            qstory (2-D array) question's indices within a story
            ---- To represent the question itself using word indices from dictionary ----
                [index of word in question, question index] = index of word in dictionary
    """
    # Try to reserve spaces beforehand (large matrices for both 1k and 10k data sets)
    # Maximum number of words in sentence = 20 # changed it to 3 as we have only 3 words in each triple
    # Maximum number of words in sentence = 10, because the object has more than 3 words some times

    '''story     = np.zeros((10, 500, len(data_files) * 3500), np.int16)
    questions = np.zeros((14, len(data_files) * 10000), np.int16)
    qstory    = np.zeros((20, len(data_files) * 10000), np.int16)'''


    # *********************************************** To be changed ****************************************************
    # Number of words in each sentence, number of sentences in each story, number of stories
    # story = np.zeros((10, 300, len(data_files) * 1), np.int16)
    story = np.zeros((1000, 1000, len(data_files) * 20), np.int16)

    # Question's meta data, number of questions
    # questions = np.zeros((14, len(data_files) * 500), np.int16)
    questions = np.zeros((14, len(data_files) * 3000), np.int16)

    # number of words in each question, number of questions
    # qstory = np.zeros((20, len(data_files) * 500), np.int16)
    qstory = np.zeros((1000, len(data_files) * 3000), np.int16)

    # NOTE: question's indices are not reset when going through a new story
    story_idx, question_idx, sentence_idx, max_words, max_sentences = -1, -1, -1, 0, 0

    # Mapping line number (within a story) to sentence's index (to support the flag include_question)
    mapping = None

    for fp in data_files:
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                line = line.rstrip().lower()
                words = line.split()

                # Story begins
                if words[0] == '1':
                    story_idx += 1
                    sentence_idx = -1
                    mapping = []

                # FIXME: This condition makes the code more fragile!
                '''if '?' not in line:
                    is_question = False
                    sentence_idx += 1
                else:
                    is_question = True
                    question_idx += 1
                    questions[0, question_idx] = story_idx
                    questions[1, question_idx] = sentence_idx
                    if include_question:
                        sentence_idx += 1'''

                if words[-1] != 'yes' and words[-1] != 'no':
                    is_question = False
                    sentence_idx += 1
                else:
                    is_question = True
                    question_idx += 1
                    questions[0, question_idx] = story_idx
                    questions[1, question_idx] = sentence_idx
                    if include_question:
                        sentence_idx += 1

                mapping.append(sentence_idx)

                # Skip substory index
                for k in range(1, len(words)):
                    w = words[k]

                    if w.endswith('.') or w.endswith('?'):
                        w = w[:-1]

                    if w not in dictionary:
                        dictionary[w] = len(dictionary)

                    if max_words < k:
                        max_words = k

                    if not is_question:
                        story[k - 1, sentence_idx, story_idx] = dictionary[w]
                    # is_question
                    else:
                        # index of each word of question
                        qstory[k - 1, question_idx] = dictionary[w]

                        '''
                        # including the question in the story
                        if include_question:
                            story[k - 1, sentence_idx, story_idx] = dictionary[w]'''

                        # NOTE: Punctuation is already removed from w
                        if k+1 ==len (words)-1:
                            if words[k+1] =='yes' or words[k+1]=='no':
                                answer = words[k + 1]
                                if answer not in dictionary:
                                    dictionary[answer] = len(dictionary)

                                questions[2, question_idx] = dictionary[answer]


                                '''
                                # Indices of supporting sentences
                                for h in range(k + 2, len(words)):
                                    questions[1 + h - k, question_idx] = mapping[int(words[h]) - 1]'''

                                questions[-1, question_idx] = line_idx
                                break

                if max_sentences < sentence_idx + 1:
                    max_sentences = sentence_idx + 1

    story     = story[:max_words, :max_sentences, :(story_idx + 1)]
    questions = questions[:, :(question_idx + 1)]
    qstory    = qstory[:max_words, :(question_idx + 1)]

    return story, questions, qstory


def build_model(general_config):
    """
    Build model

    NOTE: (for default config)
    1) Model's architecture (embedding B)
        LookupTable -> ElemMult -> Sum -> [ Duplicate -> { Parallel -> Memory -> Identity } -> AddTable ] -> LinearNB -> Softmax

    2) Memory's architecture
        a) Query module (embedding A)
            Parallel -> { LookupTable + ElemMult + Sum } -> Identity -> MatVecProd -> Softmax

        b) Output module (embedding C)
            Parallel -> { LookupTable + ElemMult + Sum } -> Identity -> MatVecProd
    """
    train_config = general_config.train_config
    dictionary   = general_config.dictionary
    use_bow      = general_config.use_bow
    nhops        = general_config.nhops
    add_proj     = general_config.add_proj
    share_type   = general_config.share_type
    #enable_time  = general_config.enable_time
    add_nonlin   = general_config.add_nonlin

    in_dim    = train_config["in_dim"]
    out_dim   = train_config["out_dim"]
    max_words = train_config["max_words"]
    voc_sz    = train_config["voc_sz"]

    if not use_bow:
        train_config["weight"] = np.ones((in_dim, max_words), np.float32)
        for i in range(in_dim):
            for j in range(max_words):
                train_config["weight"][i][j] = (i + 1 - (in_dim + 1) / 2) * \
                                               (j + 1 - (max_words + 1) / 2)
        train_config["weight"] = \
            1 + 4 * train_config["weight"] / (in_dim * max_words)

    memory = {}
    model = Sequential()
    model.add(LookupTable(voc_sz, in_dim))
    if not use_bow:

        model.add(ElemMult(train_config["weight"]))

    model.add(Sum(dim=1))

    proj = {}
    for i in range(nhops):
        if use_bow:
            memory[i] = MemoryBoW(train_config)
        else:
            memory[i] = MemoryL(train_config)

        # Override nil_word which is initialized in "self.nil_word = train_config["voc_sz"]"
        memory[i].nil_word = dictionary['nil']
        model.add(Duplicate())
        p = Parallel()
        p.add(memory[i])

        if add_proj:
            proj[i] = LinearNB(in_dim, in_dim)
            p.add(proj[i])
        else:
            p.add(Identity())

        model.add(p)
        model.add(AddTable())
        if add_nonlin:
            model.add(ReLU())


    model.add(LinearNB(out_dim, voc_sz, True))
    # Monireh added for changing the input to softmax
    model.add(LinearNB(voc_sz , 3, False))
    model.add(Identity())
    #model.add(LinearNB(out_dim, 3, True))
    model.add(Softmax())

    # Share weights
    if share_type == 1:
        # Type 1: adjacent weight tying
        memory[0].emb_query.share(model.modules[0])
        for i in range(1, nhops):
            memory[i].emb_query.share(memory[i - 1].emb_out)

        model.modules[-2].share(memory[len(memory) - 1].emb_out)

    elif share_type == 2:
        # Type 2: layer-wise weight tying
        for i in range(1, nhops):
            memory[i].emb_query.share(memory[0].emb_query)
            memory[i].emb_out.share(memory[0].emb_out)

    if add_proj:
        for i in range(1, nhops):
            proj[i].share(proj[0])

    # Cost
    loss = CrossEntropyLoss()
    loss.size_average = False
    loss.do_softmax_bprop = True
    model.modules[-1].skip_bprop = True

    return memory, model, loss

def save_model(dictionary, memory, model, loss, general_config, save_path):
    """
    This saves the trained model
    
    """
    # Get reversed dictionary mapping index to word
    reversed_dict = dict((ix, w) for w, ix in dictionary.items())

    with gzip.open(save_path, "wb") as f:
        print("Saving model to file %s ..." % save_path)
        pickle.dump((reversed_dict, model, memory, loss, general_config), f)

def load_model(model_file):
    """
    This loads in the trained model
    
    """
    print("Loading model from file %s ..." % model_file)
    with gzip.open(model_file, "rb") as f:
        reversed_dict, model, memory, loss, general_config = pickle.load(f)
    return reversed_dict, model, memory, loss, general_config


class Progress(object):
    """
    Progress bar
    """

    def __init__(self, iterable, bar_length=50):
        self.iterable = iterable
        self.bar_length = bar_length
        self.total_length = len(iterable)
        self.start_time = time.time()
        self.count = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.count += 1
            percent = self.count / self.total_length
            print_length = int(percent * self.bar_length)
            progress = "=" * print_length + " " * (self.bar_length - print_length)
            elapsed_time = time.time() - self.start_time
            print_msg = "\r|%s| %.0f%% %.1fs" % (progress, percent * 100, elapsed_time)
            sys.stdout.write(print_msg)
            if self.count == self.total_length:
                sys.stdout.write("\r" + " " * len(print_msg) + "\r")
            sys.stdout.flush()
