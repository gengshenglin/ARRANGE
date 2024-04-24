from concurrent.futures import ThreadPoolExecutor
from data.dataloader import DataLoader as DL
from data.dataloader1 import DataLoader as Dl
from data.preprocess import mnist_preprocess
from utlis.file_utls.yml_utils import read_yaml
from model.client.test import test
from model.model.resnet import ResNet, ResidualUnit
from model.model.MLP import MLP
from model.model.CNN import CNN
import matplotlib.pyplot as plt
import os
import torch


if __name__ == '__main__':
    config = read_yaml('horizontal_fl')
    data_path = config['data_path']
    client_num = config['client_num']
    lr = config['lr']
    epochs = config['epochs']
    local_epochs = config['local_epochs']
    bs = config['bs']
    gamma = config['gamma']
    dl = DL(data_path)
    minst = dl.load_horizontal_data(True, [28, 28])
    minst1 = dl.load_horizontal_data(False,[28,28])
    num_classes=10
    clients_data = mnist_preprocess(minst, client_num)
    
    times = 0
    loss_list = []
    accuracy_list = []
    idx_list = []
    epoch_test_accuracies = []  
    global_test_accuracies = [] 
    for task in task:
        times, loss, accuracy, idx,test_accuracies,all_test_accuracy= task.result()
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        idx_list.append("client-" + str(idx))
        epoch_test_accuracies.append(test_accuracies)
        global_test_accuracies.append(all_test_accuracy)
    print("train end bye!")

   
    plt.figure(1)
    x = [i for i in range(times)]
    y_list = []
    for loss in loss_list:
        y_list.append(loss)
    plt.title('loss')  
    plt.rcParams['axes.unicode_minus'] = False  
    plt.xlabel('time')
    plt.ylabel('loss')
    for y in y_list:
        plt.plot(x, y, marker='o', markersize=3)
    plt.legend(idx_list)
    
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + '50_loss_MLP_res_new.png')

    plt.figure(2)
    x = [i for i in range(times)]
    y_list = []
    for accuracy in accuracy_list:
        y_list.append(accuracy)
    plt.title('accuracy')  
    plt.rcParams['axes.unicode_minus'] = False  
    plt.xlabel('time') 
    plt.ylabel('accuracy(%)') 
    for y in y_list:
        plt.plot(x, y, marker='o', markersize=3)
    plt.legend(idx_list)  
   
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + '50_accuracy_MLP_res_new.png') 

    plt.figure(3)
    x = [i for i in range(epochs)]
    y_list = []
    for accuracy in epoch_test_accuracies:
        y_list.append(accuracy)
    plt.title('local test accuracy') 
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('epoch')
    plt.ylabel('local test accuracy(%)') 
    
    for y in y_list:
        plt.plot(x, y, marker='o', markersize=3)

    plt.legend(idx_list)
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + '50-epochs_accuracy_MLP_res_new.png')
    for y in y_list: 
        print(y)
    plt.figure(4)
    x = [i for i in range(epochs)]
    y_list =global_test_accuracies[0]
    plt.title('global test accuracy') 
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('epoch')
    plt.ylabel('global test accuracy(%)')
    plt.plot(x, y_list, marker='o', markersize=5,label='global test accuracy')
    
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + '50-global_accuracy_MLP_res_new.png') 
    print('global test accuracy:')
    print(y_list)