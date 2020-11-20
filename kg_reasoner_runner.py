import glob
import os
import random
import sys
import time
import argparse
import numpy as np

from config import KgConfig
from train_test_kg_reasoner import train, train_linear_start, test
from util_kg_reasoner import load_all_data, generate_next_story, build_model, save_model, load_model

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)  # for reproducing


def run_task(data_dir, task_name, model_file):
    """
    Train and test for each task
    """
    print("Train and test for task %s ..." % task_name)


    train_files = glob.glob('%s_train.txt' % data_dir)
    test_files = glob.glob('%s_test.txt' % data_dir)

    dictionary = {"nil": 0, "yes": 1, "no": 2}

    lines_train_data, dictionary = load_all_data(train_files, dictionary)
    lines_test_data, dictionary = load_all_data(test_files, dictionary)
    train_start = 0
    run_number = 0
    first_train = True

    print("\n#############################################################\n"
          "##################### Training Started! #####################"
          "\n#############################################################\n")

    while train_start != -1:
        # The model is trained story by story
        lines_train_data = lines_train_data[train_start:]
        train_gen = generate_next_story(lines_train_data, dictionary)
        train_story, train_questions, train_qstory, train_start = next(train_gen)
        run_number += 1

        # Very important to not make the model from scratch and remove all the trainings so far
        if first_train:
            first_train = False
            general_config = KgConfig(train_story, train_questions, dictionary)
            memory, model, loss = build_model(general_config)

        if general_config.linear_start:
            train_linear_start(train_story, train_questions, train_qstory, memory, model, loss, general_config)
        else:
            train(train_story, train_questions, train_qstory, memory, model, loss, general_config)
        if run_number % 10 == 0:
            save_path = model_file+str(run_number)
            save_model(dictionary, memory, model, loss, general_config, save_path)



    # Testing
    test_wrapper(lines_test_data, dictionary, memory, model, loss, general_config)



def run_test_task(data_dir, task_name, model_file):
    """
    Test for the task
    """
    print("testing for task %s ..." % task_name)

    test_files = glob.glob('%s_test.txt' % data_dir)
    reversed_dict, model, memory, loss, general_config = load_model(model_file)
    # Get the whole dictionary
    dictionary = dict((ix, w) for w, ix in reversed_dict.items())

    # dictionary = {"nil": 0, "yes": 1, "no": 2}
    lines_test_data, dictionary = load_all_data(test_files, dictionary)

    # Testing
    test_wrapper(lines_test_data, dictionary, memory, model, loss, general_config)


# Test wrapper, code about testing & evaluation
def test_wrapper(lines_test_data, dictionary, memory, model, loss, general_config):

    ### print("\n#############################################################\n"
    ###      "##################### Testing Started! #####################"
    ###      "\n#############################################################\n")

    test_start = 0
    test_error_total = 0.0
    precision_yes_total = 0.0
    precision_no_total = 0.0
    recall_yes_total = 0.0
    recall_no_total = 0.0
    f_measure_yes_total = 0.0
    f_measure_no_total = 0.0
    macro_avg_precision_total = 0.0
    macro_avg_recall_total = 0.0
    macro_avg_f_measure_total = 0.0

    test_count = 0
    while test_start != -1:
        lines_test_data = lines_test_data[test_start:]

        test_gen = generate_next_story(lines_test_data, dictionary)
        test_story, test_questions, test_qstory, test_start = next(test_gen)
        test_error, precision_yes, precision_no, recall_yes, recall_no, f_measure_yes, f_measure_no,macro_avg_precision,\
            macro_avg_recall, macro_avg_f_measure =\
            test(test_story, test_questions, test_qstory, memory, model, loss, general_config)

        test_error_total += test_error
        precision_yes_total += np.nan_to_num(precision_yes)  # WILL REPLACE WITH ZERO IF THE VALUE IS NAN!
        precision_no_total += np.nan_to_num(precision_no)
        recall_yes_total += np.nan_to_num(recall_yes)
        recall_no_total += np.nan_to_num(recall_no)
        f_measure_yes_total += np.nan_to_num(f_measure_yes)
        f_measure_no_total += np.nan_to_num(f_measure_no)
        macro_avg_precision_total += np.nan_to_num(macro_avg_precision)
        macro_avg_recall_total += np.nan_to_num(macro_avg_recall)
        macro_avg_f_measure_total += np.nan_to_num(macro_avg_f_measure)

        test_count += 1

    if test_count != 0:
        test_error = test_error_total / test_count
        test_precision_yes = precision_yes_total / test_count
        #test_precision_yes = np.nanmean(precision_yes_total)
        test_precision_no = precision_no_total / test_count
        #test_precision_no = np.nanmean(precision_no_total)
        test_recall_yes = recall_yes_total / test_count
        #test_recall_yes = np.nanmean(recall_yes_total)
        test_recall_no = recall_no_total / test_count
        #test_recall_no = np.nanmean(recall_no_total)
        test_f_measure_yes = f_measure_yes_total / test_count
        #test_f_measure_yes = np.nanmean(f_measure_yes_total)
        test_f_measure_no = f_measure_no_total / test_count
        #test_f_measure_no = np.nanmean(f_measure_no_total)
        test_macro_avg_precision = macro_avg_precision_total / test_count
        #test_macro_avg_precision = np.nanmean(macro_avg_precision_total)
        test_macro_avg_recall = macro_avg_recall_total / test_count
        #test_macro_avg_recall = np.nanmean(macro_avg_recall_total)
        test_macro_avg_f_measure = macro_avg_f_measure_total / test_count
        #test_macro_avg_f_measure = np.nanmean(macro_avg_f_measure_total)

        ### print(">>> Average test Error: {} <<<".format(test_error))
        ### print(">>> Average test Precision for YES class: {} <<<".format(test_precision_yes))
        ### print(">>> Average test Precision for NO class: {} <<<".format(test_precision_no))
        ### print(">>> Average test Recall for YES class: {} <<<".format(test_recall_yes))
        ### print(">>> Average test Recall for NO class: {} <<<".format(test_recall_no))
        ### print(">>> Average test F_measure for YES class: {} <<<".format(test_f_measure_yes))
        ### print(">>> Average test F_measure for NO class: {} <<<".format(test_f_measure_no))
        ### print(">>> Average test Macro Average Precision: {} <<<".format(test_macro_avg_precision))
        ### print(">>> Average test Macro Average Recall: {} <<<".format(test_macro_avg_recall))
        ### print(">>> Average test Macro Average F_measure: {} <<<".format(test_macro_avg_f_measure))
    else:
        print(">>> Test count is 0! No average test error can be calculated! <<<")
    ### print("testing finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    task_name = "sample_data_normalized"


    parser.add_argument("-d", "--data-dir", default="data/"+task_name+"/"+task_name,
                        help="path to dataset file (default: %(default)s)")
 

    parser.add_argument("-m", "--model-file", default="trained_model/"+task_name+".pklz",
                        help="model file (default: %(default)s)")

    parser.add_argument("-test", "--test", default=False, type=bool,
                    help="flag for model testing (default: %(default)s)")


    args = parser.parse_args()

    data_dir = args.data_dir

    ### print("Using data from %s" % args.data_dir)

    if args.test == False:
        start_time = time.time()
        run_task(data_dir, task_name, model_file=args.model_file)
        end_time = time.time()
        print('Total Trainig Time = {}!'.format(end_time-start_time))
    else:
        start_time = time.time()
        run_test_task(data_dir, task_name, model_file=args.model_file)
        end_time = time.time()
        print('Total Testing/Reasoning Time = {}!'.format(end_time-start_time))


