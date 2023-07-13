import attention as att
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

flags.DEFINE_float("drop", 0.9, "keep_prob for dropout")
flags.DEFINE_integer("hidden", 100, "hidden size of the model")
# TODO: modificare numero di features
flags.DEFINE_integer("num_features", 3, "keep_prob for dropout")
flags.DEFINE_integer("seq_len", 0.9, "window size of input data")

# Training parameters
flags.DEFINE_integer("num_epoch", 3000, "number of training epochs")
flags.DEFINE_integer("batch_size", 256, "batch size of training datasets")

# Paths
flags.DEFINE_string("model_name", "CAN_IDS", "name of the saved model")
flags.DEFINE_string("save_path", "../checkpoints/", "path of saved model")
# TODO: modificare path per i dati
flags.DEFINE_string("input_path", "../../data/attacks.csv", "input path of data")


# Name to change eheheh
class CANtention(object):
    """A class to build the Attention model for CAN IDS"""

    def __init__(self, num_classes, input_x, num_hidden, seq_len, keep_prob):
        super(CANtention, self).__init__()
        self.num_classes = num_classes
        self.input_x = input_x
        self.num_hidden = num_hidden
        self.seq_len = seq_len
        self.keep_prob = keep_prob

        self.positional_encoding = self.create_positional_encoding()
        self.attention_modules = self.create_attention_modules()
        self.output_layer = self.create_output_layer()

    def create_positional_encoding(self):
        return att.pos_encoding(
            self.input_x, self.num_hidden, self.seq_len, self.keep_prob
        )

    def attention_modules(self):
        return att.apply_attention(
            self.positional_encoding, self.input_x, self.num_hidden, self.seq_len
        )
