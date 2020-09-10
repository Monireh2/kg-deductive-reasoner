import numpy as np

class KgConfig(object):
    """
    Configuration for KG
    """
    def __init__(self, train_story, train_questions, dictionary):
        self.dictionary       = dictionary

        self.batch_size       = 100

        self.nhops            = 10

        # self.nepochs          = 20
        self.nepochs          = 10

        self.lrate_decay_step = 5   # reduce learning rate by half every 5 epochs

        # Use 10% of training data for validation
        self.nb_questions       = train_questions.shape[1]
        self.nb_train_questions = int(self.nb_questions * 0.9)

        self.train_range    = np.array(range(self.nb_train_questions))
        self.val_range      = np.array(range(self.nb_train_questions, self.nb_questions))
        #self.enable_time    = False   # add time embeddings # I changed it from True to False
        self.use_bow        = False  # use Bag-of-Words instead of Position-Encoding
        self.linear_start   = True
        self.share_type     = 1      # 1: adjacent, 2: layer-wise weight tying
        #self.randomize_time = 0.1    # amount of noise injected into time index # we do not need this
        self.add_proj       = False  # add linear layer between internal states ???????????????????
        self.add_nonlin     = False  # add non-linearity to internal states ???????????????????????

        if self.linear_start:

            # self.ls_nepochs          = 20
            self.ls_nepochs          = 1

            self.ls_lrate_decay_step = 5
            self.ls_init_lrate       = 0.01 / 2

        # Training configuration
        self.train_config = {
            "init_lrate"   : 0.01,
            "max_grad_norm": 40,
            "in_dim"       : 20,
            "out_dim"      : 20,
            "sz"           : min(1000, train_story.shape[1]//self.batch_size),  # number of sentences (latest sentences from the story)
            "voc_sz"       : len(self.dictionary),
            "bsz"          : self.batch_size,
            "max_words"    : len(train_story),
            "weight"       : None
        }

        if self.linear_start:
            self.train_config["init_lrate"] = 0.01 / 2
