"""
Web-based demo
"""
import glob
import flask
import numpy as np

from demo.qa_kg import MemN2N
from util_kg_reasoner import parse_kg_task, generate_next_story, load_all_data

app = flask.Flask(__name__)
memn2n = None
test_story, test_questions, test_qstory = None, None, None


def init(data_dir, model_file):
    """ Initialize web app """
    global memn2n, test_story, test_questions, test_qstory

    # Try to load model
    memn2n = MemN2N(data_dir, model_file)
    memn2n.load_model()

    # Read test data
    print("Reading test data from %s_test.txt ..." % memn2n.data_dir)
    test_data_path = glob.glob('%s_test.txt' % memn2n.data_dir)
    '''test_story, test_questions, test_qstory = \
        parse_kg_task(test_data_path, memn2n.general_config.dictionary, False)'''

    dictionary = dict((ix, w) for w, ix in memn2n.reversed_dict.items())
    lines_test_data, dictionary = load_all_data(test_data_path, dictionary)
    # print(dictionary)
    test_start = 0
    '''while test_start != -1:
        lines_test_data = lines_test_data[test_start:]

        test_gen = generate_next_story(lines_test_data, dictionary)
        test_story, test_questions, test_qstory, test_start = next(test_gen)'''
    if test_start != -1:
        lines_test_data = lines_test_data[test_start:]

        test_gen = generate_next_story(lines_test_data, dictionary)
        # print(test_gen)
        test_story, test_questions, test_qstory, test_start = next(test_gen)
        # for i in range(1000):
        #     print(test_story[:4,i,0])

def run():
    app.run()


@app.route('/')
def index():
    return flask.render_template("index.html")


@app.route('/get/story', methods=['GET'])
def get_story():
    question_idx      = np.random.randint(test_questions.shape[1])
    #story_idx         = test_questions[0, question_idx]
    story_idx = 0
    last_sentence_idx = test_questions[1, question_idx]




    story_txt, question_txt, correct_answer = memn2n.get_story_texts(test_story, test_questions, test_qstory,
                                                                     question_idx, story_idx, last_sentence_idx)
    # Format text
    story_txt = "\n".join(story_txt)
    question_txt += "?"

    return flask.jsonify({
        "question_idx": question_idx,
        "story": story_txt,
        "question": question_txt,
        "correct_answer": correct_answer
    })


@app.route('/get/answer', methods=['GET'])
def get_answer():
    question_idx  = int(flask.request.args.get('question_idx'))
    user_question = flask.request.args.get('user_question', '')

    story_idx         = test_questions[0, question_idx]
    last_sentence_idx = test_questions[1, question_idx]

    pred_answer_idx, pred_prob, memory_probs = memn2n.predict_answer(test_story, test_questions, test_qstory,
                                                                     question_idx, story_idx, last_sentence_idx,
                                                                     user_question)
    print "np.shape(memory_probs)={}".format(np.shape(memory_probs))
    #print memory_probs.T.tolist()
    pred_answer = memn2n.reversed_dict[pred_answer_idx]

    return flask.jsonify({
        "pred_answer" : pred_answer,
        "pred_prob" : pred_prob,
        "memory_probs": memory_probs.T.tolist()
    })
