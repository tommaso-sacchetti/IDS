import os
import pre_processing
import dataset_loader
import global_variables as glob
import rule_based_filtering as filter


# Set the model directory
cwd = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(cwd, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

batch_size = 128

dataset = dataset_loader.get_test_dataset()

whitelist, id_blacklist, period_blacklist, dlc_blacklist = filter.filter(dataset)

features = pre_processing.get_features(whitelist, glob.b1, glob.b2)
test_loader = dataset_loader.get_data_loader(features, batch_size)

