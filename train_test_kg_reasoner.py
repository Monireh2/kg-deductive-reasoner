from __future__ import division

import math
import numpy as np

from memn2n.nn import Softmax
from util_kg_reasoner import Progress


def train(train_story, train_questions, train_qstory, memory, model, loss, general_config):
    train_config     = general_config.train_config
    dictionary       = general_config.dictionary
    nepochs          = general_config.nepochs
    nhops            = general_config.nhops
    batch_size       = general_config.batch_size
    #enable_time      = general_config.enable_time
    #randomize_time   = general_config.randomize_time
    lrate_decay_step = general_config.lrate_decay_step

    train_range    = general_config.train_range  # indices of training questions
    val_range      = general_config.val_range    # indices of validation questions
    nb_questions   = general_config.nb_questions
    train_len      = len(train_range)
    val_len        = len(val_range)



    train_indices = np.random.choice(nb_questions, size=train_len, replace=False)
    val_indices = list(set(range(nb_questions))-set(train_indices))
    

    params = {
        "lrate": train_config["init_lrate"],
        "max_grad_norm": train_config["max_grad_norm"]
    }

    for ep in range(nepochs):
        
        print("********** Training : {} epoch **********".format(ep))
        # Decrease learning rate after every decay step
        if (ep + 1) % lrate_decay_step == 0:
            params["lrate"] *= 0.5

        total_err  = 0.
        total_cost = 0.
        total_num  = 0
        

        for _ in Progress(range(int(math.floor(train_len / batch_size)))):
            # Question batch
            batch = np.random.choice(train_indices, size=batch_size,replace=False)
            

            input_data  = np.zeros((train_story.shape[0], batch_size), np.float32) # words of training questions
            target_data = train_questions[2, batch]                                # indices of training answers
            


            memory[0].data[:] = dictionary["nil"]
            d = train_story

            # Compose batch of training data
            for b in range(batch_size):

                # NOTE: +1 since train_questions[1, :] is the index of the sentence right before the training question.
                # d is a batch of [word indices in sentence, sentence indices from batch] for this story
                input_data[:, b] = train_qstory[:, batch[b]]
            # Pick a fixed number of latest sentences (before the question) from the story
            #offset = max(0, d.shape[1] - train_config["sz"])
            #d = d[:, offset:]
            


            # Training data for the 1st memory cell
            memory[0].data[:d.shape[0], :d.shape[1]//batch_size, :] = np.squeeze(np.reshape(d,(d.shape[0],-1,batch_size)))
            

            for i in range(1, nhops):
                memory[i].data = memory[0].data
            
            out = model.fprop(input_data)
            
            total_cost += loss.fprop(out, target_data)
            total_err  += loss.get_error(out, target_data)
            total_num  += batch_size

            grad = loss.bprop(out, target_data)
            model.bprop(input_data, grad)
            model.update(params)

            for i in range(nhops):
                memory[i].emb_query.weight.D[:, 0] = 0

        if total_num == 0:
            print ("Train size smaller than batch size!  Give up current task & continued!")
            continue

        # Validation
        total_val_err  = 0.
        total_val_cost = 0.
        total_val_num  = 0

        for k in range(int(math.floor(val_len / batch_size))):
            batch = np.random.choice(val_indices, size=batch_size, replace=False)
            input_data  = np.zeros((train_story.shape[0], batch_size), np.float32)
            target_data = train_questions[2, batch]

            memory[0].data[:] = dictionary["nil"]
            d = train_story
            #offset = max(0, d.shape[1] - train_config["sz"])
            #d = d[:, offset:]

            # Data for the 1st memory cell
            memory[0].data[:d.shape[0], :d.shape[1] // batch_size, :] = np.squeeze(
                np.reshape(d, (d.shape[0], -1, batch_size)))
            #memory[0].data[:d.shape[0], :d.shape[1], b] = np.squeeze(d)
            for b in range(batch_size):
                #d = train_story[:, :(1 + train_questions[1, batch[b]]), train_questions[0, batch[b]]]




                input_data[:, b] = train_qstory[:, batch[b]]

            for i in range(1, nhops):
                memory[i].data = memory[0].data

            out = model.fprop(input_data)
            total_val_cost += loss.fprop(out, target_data)
            total_val_err  += loss.get_error(out, target_data)
            total_val_num  += batch_size

        if total_val_num == 0:
            print ("Val size smaller than batch size!  Give up current task & continued!")
            continue

        train_error = total_err / total_num
        val_error   = total_val_err / total_val_num

        print("%d | train error: %g | val error: %g" % (ep + 1, train_error, val_error))


def train_linear_start(train_story, train_questions, train_qstory, memory, model, loss, general_config):

    train_config = general_config.train_config

    # Remove softmax from memory
    for i in range(general_config.nhops):
        memory[i].mod_query.modules.pop()

    # Save settings
    nepochs2          = general_config.nepochs
    lrate_decay_step2 = general_config.lrate_decay_step
    init_lrate2       = train_config["init_lrate"]

    # Add new settings
    general_config.nepochs          = general_config.ls_nepochs
    general_config.lrate_decay_step = general_config.ls_lrate_decay_step
    train_config["init_lrate"]      = general_config.ls_init_lrate

    # Train with new settings
    train(train_story, train_questions, train_qstory, memory, model, loss, general_config)

    # Add softmax back
    for i in range(general_config.nhops):
        memory[i].mod_query.add(Softmax())

    # Restore old settings
    general_config.nepochs          = nepochs2
    general_config.lrate_decay_step = lrate_decay_step2
    train_config["init_lrate"]      = init_lrate2

    # Train with old settings
    train(train_story, train_questions, train_qstory, memory, model, loss, general_config)


def test(test_story, test_questions, test_qstory, memory, model, loss, general_config):

    total_test_err = 0.
    total_test_num = 0
    total_tp_yes = 0.0
    total_fp_yes = 0.0
    total_fn_yes = 0.0
    total_tp_no = 0.0
    total_fp_no = 0.0
    total_fn_no = 0.0
    precision_yes_arr = []
    recall_yes_arr = []
    precision_no_arr = []
    recall_no_arr = []

    nhops        = general_config.nhops
    train_config = general_config.train_config
    batch_size   = general_config.batch_size
    dictionary   = general_config.dictionary

    '''print "############################################################################################################"
    for key,value in dictionary.items():
        print key,value
    print "############################################################################################################"'''

    max_words = train_config["max_words"]

    for k in range(int(math.floor(test_questions.shape[1] / batch_size))):
        batch = np.arange(k * batch_size, (k + 1) * batch_size)

        input_data = np.zeros((max_words, batch_size), np.float32)
        target_data = test_questions[2, batch]



        input_data[:]     = dictionary["nil"]
        memory[0].data[:] = dictionary["nil"]
        d = test_story
        #offset = max(0, d.shape[1] - train_config["sz"])
        #d = d[:, offset:]

        #memory[0].data[:d.shape[0], :d.shape[1], b] = np.squeeze(d)
        memory[0].data[:d.shape[0], :d.shape[1] // batch_size, :] = np.squeeze(
            np.reshape(d, (d.shape[0], -1, batch_size)))

        for b in range(batch_size):
            #d = test_story[:, :(1 + test_questions[1, batch[b]]), test_questions[0, batch[b]]]

            
            input_data[:test_qstory.shape[0], b] = test_qstory[:, batch[b]]

        for i in range(1, nhops):
            memory[i].data = memory[0].data

        out = model.fprop(input_data)
        cost = loss.fprop(out, target_data)


        total_test_err += loss.get_error(out, target_data)

        batch_precision_yes, batch_recall_yes, batch_tp_yes, batch_fp_yes, batch_fn_yes = loss.get_precision_recall_yes\
            (out, target_data)
        batch_precision_no, batch_recall_no, batch_tp_no, batch_fp_no, batch_fn_no = loss.get_precision_recall_no\
            (out, target_data)

        precision_yes_arr.append(batch_precision_yes)
        recall_yes_arr.append(batch_recall_yes)
        precision_no_arr.append(batch_precision_no)
        recall_no_arr.append(batch_recall_no)

        total_tp_yes += batch_tp_yes
        total_fp_yes += batch_fp_yes
        total_fn_yes += batch_fn_yes
        total_tp_no += batch_tp_no
        total_fp_no += batch_fp_no
        total_fn_no += batch_fn_no


        total_test_num += batch_size

        ### print "precision for yes=" + str(batch_precision_yes)
        ### print "recall for yes=" + str(batch_recall_yes)
        ### print "precision for no=" + str(batch_precision_no)
        ### print "recall for no=" + str(batch_recall_no)




        pred_answer_idx = out.argmax(axis=0)
        pred_prob = out[pred_answer_idx,0]
        ### for i,(a,t) in enumerate(zip(pred_answer_idx,target_data)):

            ### print("predicted result: {:3}, probability: {:3}, target: {}".format(a, pred_prob[i],t))
    if total_test_num == 0:
        print("Test size smaller than batch size! Give up current task.")
        return

    test_error = total_test_err / total_test_num
    precision_yes_over_batches = np.nanmean(precision_yes_arr) # Problem: nans
    precision_no_over_batches = np.nanmean(precision_no_arr)

    recall_yes_over_batches = np.nanmean(recall_yes_arr)
    recall_no_over_batches = np.nanmean(recall_no_arr)

    precision_yes = np.nan
    recall_yes = np.nan
    precision_no = np.nan
    recall_no = np.nan
    f_measure_yes = np.nan
    f_measure_no = np.nan
    macro_avg_precision = np.nan
    macro_avg_recall = np.nan
    macro_avg_f_measure = np.nan
    print("\n>>> Test error for current kg: {} <<<\n".format(test_error))

    try:
        precision_yes = total_tp_yes/(total_tp_yes + total_fp_yes)
        recall_yes = total_tp_yes/(total_tp_yes + total_fn_yes)
        precision_no = total_tp_no/(total_tp_no + total_fp_no)
        recall_no = total_tp_no/(total_tp_no + total_fn_no)
        ### print("\n>>> Precision for Yes class for current kg: {} <<<\n".format(precision_yes))
        ### print("\n>>> Recall for Yes class for current kg: {} <<<\n".format(recall_yes))
        ### print("\n>>> Precision for No class for current kg: {} <<<\n".format(precision_no))
        ### print("\n>>> Recall for No class for current kg: {} <<<\n".format(recall_no))
        macro_avg_precision = (precision_yes + precision_no)/2
        macro_avg_recall = (recall_yes + recall_no)/2
        f_measure_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes)
        f_measure_no = 2 * precision_no * recall_no / (precision_no + recall_no)



        macro_avg_f_measure = 2 * macro_avg_precision * macro_avg_recall / (macro_avg_precision + macro_avg_recall)



        print("\n>>> F-Measure for Yes class for current kg: {} <<<\n".format(f_measure_yes))

        print("\n>>> F-Measure for No class for current kg: {} <<<\n".format(f_measure_no))
        print("\n>>> Macro Average F-Measure for current kg: {} <<<\n".format(macro_avg_f_measure))

    except ZeroDivisionError:
        print ("ZeroDivisionError")

    finally:
        #  print test_error, precision_yes, precision_no, recall_yes, recall_no
        return test_error, precision_yes, precision_no, recall_yes, recall_no, f_measure_yes, f_measure_no, \
           macro_avg_precision, macro_avg_recall, macro_avg_f_measure
