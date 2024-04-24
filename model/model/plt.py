import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    plt.figure(3)
    x = [i for i in range(30)]
    y_list = []
    epoch_test_accuracies = []
    global_test_accuracies = []
    for accuracy in epoch_test_accuracies:
        y_list.append(accuracy)
    plt.title('test accuracy(%)')
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    for y in y_list:
        plt.plot(x, y, marker='o', markersize=3)
    plt.legend(0)
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + 'NON-iid-epochs_accuracy_CNN_res_new.png') 
    plt.figure(4)
    x = [i for i in range(30)]
    y_list =global_test_accuracies
    plt.title('global test accuracy')
    plt.rcParams['axes.unicode_minus'] = False 
    plt.xlabel('epoch')
    plt.ylabel('global test accuracy(%)')
    plt.plot(x, y, marker='o', markersize=3,label='Global test accuracy')
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + 'NON-iid-global_accuracy_CNN_res_new.png')