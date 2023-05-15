# Deep-Learning based Intrusion Detection System for In-Vehicle Controller Area Network
Msc thesis in Computer Science and Engineering. The aim of the project is to build, train and test the state-of-the-art deep learning model that perform intrusion detection on automotive controller area networks.
The models being implemented are four, they are all based on papers and they use different deep learning techniques:
* (Hybrid) Deep Neural Network, based on "A Hybrid Approach Toward Efficient and Accurate Intrusion Detection for In-Vehicle Networks" - Zhang et al.
* Convolutional Neural Network, based on "An Effective In-Vehicle CAN Bus Intrusion Detection System Using CNN Deep Learning Approach", - Hossain et al.
* Attention technique, based on "In-Vehicle Intrusion Detection Based on Deep Learning Attention Technique", - NasrEldin et al.
* LSTM, based on "MLIDS: Handling Raw High-Dimensional CAN Bus Data Using Long Short-Term Memory Networks for Intrusion Detection in In-Vehicle Networks", Desta et al.

## Hybrid Deep Neural Network
The system is an hybrid intrusion detection system with a rule-based module and a deep learning model trained to recognize malicious CAN packets.
The rule-based implemented checks for valid IDs, valid DLCs and if the packets periodicity is respected.

The model takes as features tha ids, two bytes of the payload, the payload's entropy and the hamming distance from the last packet of the same ID sent.

_development ongoing_

## Convolutional Neural Network
The network is a 1-D convolutional network taking as input the 8 bytes of the payload as integers, the IDs of the packets and the flag column.
Both a multiclass and a binary classification will be tested.

_development ongoing_