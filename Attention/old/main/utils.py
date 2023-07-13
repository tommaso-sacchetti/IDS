import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def weighted_crossentropy(y_true, y_pred):
    output = y_pred
    output /= torch.sum(y_pred, dim=-1, keedim=True)
    output = torch.clamp(output( 1e-5, 1. - 1e-5))

    pos_weight = torch.sum(y_true[:, :, 0]) / (torch.sum(y_true[:, :, 1]) + 1e-5)

    loss = y_true[:, :, 1] * pos_weight * torch.log(output[:, :, 1]) + y_true[:, :, 0] * 1 * torch.log(output[:, :, 0])
    xent = -torch.mean(loss)

    return xent


def evaluate_test(testX, testY, Input_X, Input_Y, keep_prob, model, loss_fn, batch_size):
    '''Function used to do inference on validation and test datasets'''
    # TODO: check if function is correct 
    
    num_test_samples = testX.shape[0]
    num_test_batch = (num_test_samples + batch_size - 1) // batch_size

    test_batch_loss = 0
    test_pred_init = None

    for batch in range(num_test_batch):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, num_test_samples)

        mini_test_x = torch.from_numpy(testX[start_idx:end_idx])
        mini_test_y = torch.from_numpy(testY[start_idx:end_idx])

        model.eval()
        with torch.no_grad():
            test_pred = model(mini_test_x, mini_test_y, keep_prob=1.0)
            test_loss = loss_fn(test_pred, mini_test_y)

        test_batch_loss += test_loss.item()

        if test_pred_init is None:
            test_pred_init = test_pred
        else:
            test_pred_init = torch.cat([test_pred_init, test_pred], dim=0)

    test_batch_loss /= num_test_batch

    return test_batch_loss, test_pred_init


def get_result(test_pred, testY, threshold):
    """ Function used to compute F1 scores """

    pred = test_pred.reshape(-1, test_pred.shape[2]) 
    pred = (pred > threshold).int()

    targ = testY.reshape(-1, testY.shape[2])

    pred_attack = pred[:, 1]
    targ_attack = targ[:, 1]
    conf = confusion_matrix(targ_attack, pred_attack)

    _test_precision, _test_recall, _test_f1, _support = precision_recall_fscore_support(targ, pred)

    return _test_precision, _test_recall, _test_f1, _support, pred, targ, conf


def plot_performance(train_loss, val_loss, train_f1, val_f1, title):

	labels = [ "train_loss", "val_loss", "train_f1", "val_f1" ]
	
	plot_data = dict()
	plot_data["train_loss"] = train_loss
	plot_data["val_loss"] = val_loss
	plot_data["train_f1"] = train_f1
	plot_data["val_f1"] = val_f1
	
	
	fig, ax = plt.subplots(figsize = (10,8))
	for metric in labels:
		ax.plot( np.arange(len(plot_data[metric])) , np.array(plot_data[metric]), label=metric)
	ax.legend()
	ax.set_ylim(0.0, 1)
	plt.title(title)
	plt.xlabel('no. of epoch')
	fig.savefig('../plot_train/' + title + '.png')
	#plt.show()


def shuffle_fixed_data(data, label):
    # TODO: check full compatibility with pytorch
	idx = np.arange(len(data)) #create array of index for data
	np.random.shuffle(idx) #randomly shuffle idx
	data = data[idx]
	label = label[idx]
	
	return data, label


def get_total_para(trainable_parameters):
    total_parameters = 0

    for parameter in trainable_parameters:
        shape = parameter.shape
        variable_parameters = 1

        for dim in shape: 
            variable_parameters *= dim

        total_parameters += variable_parameters

    print('total_parameters:', total_parameters)


def test_sample(model, testX, Input_X, keep_prob):
    '''Function used to compute inference time of a single sample'''
    
    one_sample = testX[[0], :, :]
    past = datetime.now()
    
    model.eval()
    with torch.no_grad():
        prediction = model(one_sample, keep_prob=1.0)
    
    print("test time one sample {:.5f}s".format((datetime.now() - past).total_seconds()))



