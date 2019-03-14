export THEANO_FLAGS='device=cuda1, gpuarray.preallocate=.1'
CODEFOLDER="/home/btek/Dropbox/code/pythoncode/FocusingNeuron/"
EXFOLDER="/home/btek/Dropbox/code/pythoncode/FocusingNeuron/outputs/ESNN/mnist9/mlp"
EXNAME="mnist9/mlp/"
RUN_SCRIPT="mnist.py"
echo $CODEFOLDER$RUN_SCRIPT
#cd $EXFOLDER

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT mlp:2,800,.25,.50 350 5 mnist $EXNAME 0.0 >> $EXFOLDER/mnist_mlp_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/mnist_result_mlp* >> $EXFOLDER/list_mnist_mlp.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/mnist_mlp_log.txt)
printf "MLP\n" >> $EXFOLDER/mnist_mlp_accuracy_log.txt
date >> $EXFOLDER/mnist_mlp_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/mnist_mlp_accuracy_log.txt


# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT mlp_mlp:2,800,.25,.50 350 5 fashion $EXNAME 0.0 >> $EXFOLDER/fashion_mlp_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/fashion_result_mlp* >> $EXFOLDER/list_fashion_mlp.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/fashion_mlp_log.txt)
printf "MLP\n" >> $EXFOLDER/fashion_mlp_accuracy_log.txt
date >> $EXFOLDER/fashion_mlp_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/fashion_mlp_accuracy_log.txt

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT mlp_mlp:2,800,.25,.50 350 5 cifar10 $EXNAME 0.0 >> $EXFOLDER/cifar10_mlp_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/cifar10_result_mlp* >> $EXFOLDER/list_cifar10_mlp.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/cifar10_mlp_log.txt)
printf "MLP\n" >> $EXFOLDER/cifar10_mlp_accuracy_log.txt
date >> $EXFOLDER/cifar10_mlp_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/cifar10_mlp_accuracy_log.txt

# RUN THE PYTHON SCRIPT
python $CODEFOLDER$RUN_SCRIPT mlp_mlp:2,1200,.25,.50 350 5 mnist_cluttered $EXNAME 0.0 >> $EXFOLDER/mnist_cluttered_mlp_log.txt
# LIST THE OUTPUT FILES
ls $EXFOLDER/mnist_cluttered_result_mlp* >> $EXFOLDER/list_mnist_cluttered_mlp.txt
# FIND TEST ACCURACIES WARNING THIS IS JUST A PRINT SUMMARY. ALL TEST ACCURACIES ARE ACTUALLY IN NPZ FILE
SUMMARY=$(grep "test accuracy:" $EXFOLDER/mnist_cluttered_mlp_log.txt)
printf "MLP\n" >> $EXFOLDER/mnist_cluttered_mlp_accuracy_log.txt
date >> $EXFOLDER/mnist_cluttered_mlp_accuracy_log.txt
echo $SUMMARY>> $EXFOLDER/mnist_cluttered_mlp_accuracy_log.txt
