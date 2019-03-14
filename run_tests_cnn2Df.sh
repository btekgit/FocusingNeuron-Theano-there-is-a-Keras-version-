export THEANO_FLAGS='device=cuda0, floatX=float32, gpuarray.preallocate=.1'
CODEFOLDER="/home/btek/Dropbox/code/pythoncode/FocusingNeuron/"
EXFOLDER="/home/btek/Dropbox/code/pythoncode/FocusingNeuron/outputs/ESNN/cnn/mar12/2DD"
EXNAME="cnn/mar12/2DD/"
RUN_SCRIPT="mnist.py"
NUM_EPOCHS=50
NUM_REPEATS=5
echo $CODEFOLDER$RUN_SCRIPT
#cd $EXFOLDER

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_cnn $NUM_EPOCHS $NUM_REPEATS mnist $EXNAME 0.0 >> $EXFOLDER/mnist_focused_cnn_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/mnist_result_cnn* >> $EXFOLDER/list_mnist_focused_cnn.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/mnist_focused_cnn_log.txt)
printf "focused_cnn\n" >> $EXFOLDER/mnist_focused_cnn_accuracy_log.txt
date >> $EXFOLDER/mnist_focused_cnn_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/mnist_focused_cnn_accuracy_log.txt


# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_cnn $NUM_EPOCHS $NUM_REPEATS fashion $EXNAME 0.25 >> $EXFOLDER/fashion_focused_cnn_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/fashion_result_focused_cnn* >> $EXFOLDER/list_fashion_focused_cnn.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/fashion_focused_cnn_log.txt)
printf "CNN\n" >> $EXFOLDER/fashion_focused_cnn_accuracy_log.txt
date >> $EXFOLDER/fashion_focused_cnn_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/fashion_focused_cnn_accuracy_log.txt

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_cnn $NUM_EPOCHS $NUM_REPEATS cifar10 $EXNAME 0.25 >> $EXFOLDER/cifar10_focused_cnn_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/cifar10_result_focused_cnn* >> $EXFOLDER/list_cifar10_focused_cnn.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/cifar10_focused_cnn_log.txt)
printf "CNN\n" >> $EXFOLDER/cifar10_focused_cnn_accuracy_log.txt
date >> $EXFOLDER/cifar10_focused_cnn_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/cifar10_focused_cnn_accuracy_log.txt

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_cnn $NUM_EPOCHS $NUM_REPEATS mnist_cluttered $EXNAME 0.25 >> $EXFOLDER/mnist_cluttered_focused_cnn_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/mnist_cluttered_result_focused_cnn* >> $EXFOLDER/list_mnist_cluttered_focused_cnn.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/mnist_cluttered_focused_cnn_log.txt)
printf "CNN\n" >> $EXFOLDER/mnist_cluttered_focused_cnn_accuracy_log.txt
date >> $EXFOLDER/mnist_cluttered_focused_cnn_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/mnist_cluttered_focused_cnn_accuracy_log.txt
