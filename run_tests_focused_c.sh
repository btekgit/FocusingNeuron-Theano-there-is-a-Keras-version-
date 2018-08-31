source activate theano3
export THEANO_FLAGS='device=cuda0, gpuarray.preallocate=.1'
CODEFOLDER="/home/btek/Dropbox/code/pythoncode/FocusingNeuron/"
EXFOLDER="/home/btek/Dropbox/code/pythoncode/FocusingNeuron/outputs/ESNN/mnist9/focused_c"
EXNAME="mnist9/focused_c/"
RUN_SCRIPT="mnist_tmp.py"
echo $CODEFOLDER$RUN_SCRIPT
#cd $EXFOLDER

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_mlp:2,800,.25,.50 350 5 mnist $EXNAME 2.0 >> $EXFOLDER/mnist_focused_c_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/mnist_result_focused* >> $EXFOLDER/list_mnist_focused_c.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/mnist_focused_c_log.txt)
echo focused>> $EXFOLDER/mnist_focused_c_accuracy_log.txt
date >> $EXFOLDER/mnist_focused_c_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/mnist_focused_c_accuracy_log.txt


# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_mlp:2,800,.25,.50 350 5 fashion $EXNAME 0.0 >> $EXFOLDER/fashion_focused_c_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/fashion_result_focused* >> $EXFOLDER/list_fashion_focused_c.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/fashion_focused_c_log.txt)
echo focused_c_>> $EXFOLDER/fashion_focused_c_accuracy_log.txt
date >> $EXFOLDER/fashion_focused_c_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/fashion_focused_c_accuracy_log.txt

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_mlp:2,800,.25,.50 350 5 cifar10 $EXNAME 0.0 >> $EXFOLDER/cifar10_focused_c_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/cifar10_result_focused* >> $EXFOLDER/list_cifar10_focused.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/cifar10_focused_c_log.txt)
echo focused>> $EXFOLDER/cifar10_focused_c_accuracy_log.txt
date >> $EXFOLDER/cifar10_focused_c_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/cifar10_focused_c_accuracy_log.txt

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT focused_mlp:2,1200,.25,.50 350 5 mnist_cluttered $EXNAME 0.0 >> $EXFOLDER/mnist_cluttered_focused_c_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/mnist_cluttered_result_focused* >> $EXFOLDER/list_mnist_cluttered_focused.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/mnist_cluttered_log.txt)
echo focused>> $EXFOLDER/mnist_cluttered_focused_c_accuracy_log.txt
date >> $EXFOLDER/mnist_cluttered_focused_c_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/mnist_cluttered_focused_c_accuracy_log.txt

