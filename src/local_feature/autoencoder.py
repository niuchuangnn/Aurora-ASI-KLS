'''
Chuang Niu, niuchuang@stu.xidian.edu.cn
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../fast-rcnn/caffe-fast-rcnn/python')
import caffe
from caffe import layers as L, params as P
import os
import h5py
from caffe.proto import caffe_pb2
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf


def solver(train_net_path, test_net_path, paraConfig,
           save_path='../../Data/autoEncoder/auto_encoder_solver.prototxt'):
    s = caffe_pb2.SolverParameter()

    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = paraConfig['test_interval']
        s.test_iter.append(paraConfig['test_iter'])

    s.type = paraConfig['type']
    s.base_lr = paraConfig['base_lr']
    s.lr_policy = paraConfig['lr_policy']
    s.gamma = paraConfig['gamma']
    s.stepsize = paraConfig['stepsize']
    s.momentum = paraConfig['momentum']
    s.weight_decay = paraConfig['weight_decay']
    s.max_iter = paraConfig['max_iter']
    s.display = paraConfig['display']
    s.snapshot = paraConfig['snapshot']
    s.snapshot_prefix = paraConfig['snapshot_prefix']+paraConfig['layer_idx']
    s.solver_mode = paraConfig['solver_mode']

    with open(save_path, 'w') as f:
        f.write(str(s))
    return save_path




def layerwise_train_net(layer_idx, paraConfig, isTrain):
    """define layer-wise training net, return the resulting protocol buffer file,
    use str function can turn this result to text file.
    """
    n = caffe.NetSpec()

    if isTrain:
        batch_size = paraConfig['train_batchSize']
        h5 = paraConfig['train_data']
    else:
        batch_size = paraConfig['test_batchSize']
        h5 = paraConfig['test_data']

    n.data, n.label = L.HDF5Data(source=h5, batch_size=batch_size, shuffle=True, ntop=2)
    flatdata = L.Flatten(n.data)
    flatdata_name = 'flatdata'
    n.__setattr__(flatdata_name, flatdata)
    drop_ratio = paraConfig['drop_ratio']

    for l in range(layer_idx + 1):
        if l == layer_idx:
            param = paraConfig['learned_param']
        else:
            param = paraConfig['frozen_param']

        if l == 0:
            if layer_idx == 0:
                drop_noise = L.Dropout(n.flatdata, dropout_param=dict(dropout_ratio=drop_ratio), in_place=False)
                drop_noise_name = 'drop_noise' + str(l + 1)
            else:
                drop_noise = flatdata
                drop_noise_name = flatdata_name

        if l > 0:
            if l == layer_idx:
                drop_noise = L.Dropout(n[relu_en_name], dropout_param=dict(dropout_ratio=drop_ratio), in_place=False)
                drop_noise_name = 'drop_noise' + str(l + 1)
            else:
                drop_noise = relu_en
                drop_noise_name = relu_en_name

        n.__setattr__(drop_noise_name, drop_noise)

        encoder = L.InnerProduct(n[drop_noise_name], num_output=layerNeuronNum[l+1], param=param,
                                 weight_filler=dict(type='gaussian', std=0.005),
                                 bias_filler=dict(type='constant', value=0.1))
        encoder_name = 'encoder' + str(l + 1)
        n.__setattr__(encoder_name, encoder)

        relu_en = L.ReLU(n[encoder_name], in_place=True)
        relu_en_name = 'relu_en' + str(l + 1)
        n.__setattr__(relu_en_name, relu_en)


        if l == layer_idx:
            drop_en = L.Dropout(n[relu_en_name], dropout_param=dict(dropout_ratio=drop_ratio), in_place=True)
            drop_en_name = 'drop_en' + str(l + 1)
            n.__setattr__(drop_en_name, drop_en)

            decoder = L.InnerProduct(n[drop_en_name], num_output=layerNeuronNum[l], param=param,
                                     weight_filler=dict(type='gaussian', std=0.005),
                                     bias_filler=dict(type='constant', value=0.1))
            decoder_name = 'decoder' + str(l + 1)
            n.__setattr__(decoder_name, decoder)

            if l > 0:
                relu_de = L.ReLU(n[decoder_name], dropout_param=dict(dropout_ratio=drop_ratio), in_place=True)
                relu_de_name = 'relu_de' + str(l + 1)
                n.__setattr__(relu_de_name, relu_de)

                n.loss = L.EuclideanLoss(n[relu_de_name], n['relu_en' + str(l)])
            if l == 0:
                n.loss = L.EuclideanLoss(n[decoder_name], n.flatdata)

    return n.to_proto()

def finetuningNet(h5, batch_size, layerNum):
    n = caffe.NetSpec()

    n.data, n.label = L.HDF5Data(source=h5, batch_size=batch_size, shuffle=True, ntop=2)
    flatdata = L.Flatten(n.data)
    flatdata_name = 'flatdata'
    n.__setattr__(flatdata_name, flatdata)

    param = learned_param
    for l in range(layerNum):
        if l == 0:
            encoder_name_last = flatdata_name
        else:
            encoder_name_last = relu_en_name

        encoder = L.InnerProduct(n[encoder_name_last], num_output=layerNeuronNum[l + 1], param=param,
                                 weight_filler=dict(type='gaussian', std=0.005),
                                 bias_filler=dict(type='constant', value=0.1))
        encoder_name = 'encoder' + str(l + 1)
        n.__setattr__(encoder_name, encoder)

        relu_en = L.ReLU(n[encoder_name], in_place=True)
        relu_en_name = 'relu_en' + str(l + 1)
        n.__setattr__(relu_en_name, relu_en)

    for l in range(layerNum):
        if l == 0:
            decoder_name_last = relu_en_name
        else:
            decoder_name_last = relu_de_name

        decoder = L.InnerProduct(n[decoder_name_last], num_output=layerNeuronNum[layerNum-l-1], param=param,
                                 weight_filler=dict(type='gaussian', std=0.005),
                                 bias_filler=dict(type='constant', value=0.1))
        decoder_name = 'decoder' + str(layerNum - l)
        n.__setattr__(decoder_name, decoder)

        if l < (layerNum-1):
            relu_de = L.ReLU(n[decoder_name], in_place=True)
            relu_de_name = 'relu_de' + str(layerNum-l)
            n.__setattr__(relu_de_name, relu_de)

        n.loss = L.EuclideanLoss(n[decoder_name], n.flatdata)

    return n.to_proto()

def defineTestNet(inputShape, layerNeuronNum):
    layerNum = len(layerNeuronNum) - 1
    n = caffe.NetSpec()

    # n.data = L.Input(input_param=dict(shape=inputShape))
    n.data, n.label = L.MemoryData(memory_data_param=dict(batch_size=inputShape[0], channels=inputShape[1],
                                                 height=inputShape[2], width=inputShape[3]), ntop=2)
    flatdata = L.Flatten(n.data)
    flatdata_name = 'flatdata'
    n.__setattr__(flatdata_name, flatdata)

    for l in range(layerNum):
        if l == 0:
            encoder_name_last = flatdata_name
        else:
            encoder_name_last = relu_en_name

        encoder = L.InnerProduct(n[encoder_name_last], num_output=layerNeuronNum[l + 1])
        encoder_name = 'encoder' + str(l + 1)
        n.__setattr__(encoder_name, encoder)

        relu_en = L.ReLU(n[encoder_name], in_place=True)
        relu_en_name = 'relu_en' + str(l + 1)
        n.__setattr__(relu_en_name, relu_en)

    return n.to_proto()

def classificationNet(h5, batch_size, layerNeuronNum, layerNum, classNum, learned_param):
    n = caffe.NetSpec()

    n.data, n.label = L.HDF5Data(source=h5, batch_size=batch_size, shuffle=True, ntop=2)
    flatdata = L.Flatten(n.data)
    flatdata_name = 'flatdata'
    n.__setattr__(flatdata_name, flatdata)

    param = learned_param
    for l in range(layerNum):
        if l == 0:
            encoder_name_last = flatdata_name
        else:
            encoder_name_last = relu_en_name

        encoder = L.InnerProduct(n[encoder_name_last], num_output=layerNeuronNum[l + 1], param=param,
                                 weight_filler=dict(type='gaussian', std=0.005),
                                 bias_filler=dict(type='constant', value=0.1))
        encoder_name = 'encoder' + str(l + 1)
        n.__setattr__(encoder_name, encoder)

        relu_en = L.ReLU(n[encoder_name], in_place=True)
        relu_en_name = 'relu_en' + str(l + 1)
        n.__setattr__(relu_en_name, relu_en)

    output = L.InnerProduct(n[relu_en_name], num_output=classNum, param=param,
                            weight_filler=dict(type='gaussian', std=0.005),
                            bias_filler=dict(type='constant', value=0.1))
    output_name = 'output'
    n.__setattr__(output_name, output)

    n.loss = L.SoftmaxWithLoss(n[output_name], n.label)

    return n.to_proto()

def layerwise_train(paraConfig):
    layerNeuronNum = paraConfig['layerNeuronNum']
    layerNum = len(layerNeuronNum) - 1
    for layer_idx in range(layerNum):
        print 'layerwise_train: ' + str(layer_idx+1)
        train_proto = 'autoencoder_train' + str(layer_idx+1) + '.prototxt'
        test_proto = 'autoencoder_test' + str(layer_idx+1) + '.prototxt'
        solver_proto = 'auto_encoder_solver' + str(layer_idx+1) + '.prototxt'
        autoSaveFolder = paraConfig['autoSaveFolder']

        with open(autoSaveFolder+train_proto, 'w') as f_tr:
            f_tr.write(str(layerwise_train_net(layer_idx, paraConfig, True)))

        with open(autoSaveFolder+test_proto, 'w') as f_te:
            f_te.write(str(layerwise_train_net(layer_idx, paraConfig, False)))

        snapshot_prefix = paraConfig['snapshot_prefix'] + str(layer_idx+1)
        paraConfig['layer_idx'] = str(layer_idx+1)
        solver_proto = solver(autoSaveFolder+train_proto, autoSaveFolder+test_proto, paraConfig, save_path=autoSaveFolder+solver_proto)

        caffe.set_device(0)
        caffe.set_mode_gpu()

        caffe_solver = None
        caffe_solver = caffe.SGDSolver(solver_proto)

        if layer_idx != 0:
            caffe_solver.net.copy_from(autoSaveFolder+paraConfig['snapshot_prefix']+str(layer_idx)+'.caffemodel')

        niter = paraConfig['max_iter']
        test_interval = paraConfig['test_interval']
        # losses will also be stored in the log
        train_loss = np.zeros((niter,))
        test_loss = np.zeros((niter // test_interval,))

        isTest = paraConfig['isTest']
        # the main solver loop
        for it in range(niter):
            caffe_solver.step(1)  # SGD by Caffe

            # store the train loss
            train_loss[it] = caffe_solver.net.blobs['loss'].data

            # run a full test every so often
            # (Caffe can also do this for us and write to a log, but we show here
            #  how to do it directly in Python, where more complicated things are easier.)
            if isTest:
                if it % test_interval == 0:
                    print 'Iteration', it, 'testing...'
                    loss_test = 0
                    for test_it in range(paraConfig['test_iter']):
                        caffe_solver.test_nets[0].forward()
                        loss_test += np.sum(caffe_solver.test_nets[0].blobs['loss'].data)

                    test_loss[it // test_interval] = loss_test / (paraConfig['test_iter']*paraConfig['test_batchSize'])

        caffe_solver.net.save(autoSaveFolder+snapshot_prefix+'.caffemodel')
        print autoSaveFolder+snapshot_prefix+'.caffemodel'+' saved'
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(np.arange(niter), train_loss / paraConfig['train_batchSize'])
        ax2.plot(test_interval * np.arange(int(np.ceil(niter / test_interval))),
                 test_loss[0:int(np.ceil(niter / test_interval))], 'r')

        plt.title('layer'+str(layer_idx+1))
        plt.draw()

        print test_loss[0], test_loss[int(np.ceil(niter / test_interval)) - 1]
        print train_loss[0] / 256, train_loss[-1] / 256
    return 0

def finetue_train(paraConfig, finetune_train_proto= '../../Data/autoEncoder/finetune_train.prototxt',
                  finetune_test_proto='../../Data/autoEncoder/finetune_test.prototxt',
                  finetune_solver_proto='../../Data/autoEncoder/finetune_solver.prototxt'):
    train_data = paraConfig['train_data']
    test_data = paraConfig['test_data']
    autoSaveFolder = paraConfig['autoSaveFolder']
    layerNeuronNum = paraConfig['layerNeuronNum']
    layerNum = len(layerNeuronNum) - 1
    train_batchSize = paraConfig['train_batchSize']
    test_batchSize = paraConfig['test_batchSize']
    snapshot_prefix = paraConfig['snapshot_prefix']

    with open(finetune_train_proto, 'w') as f1:
        f1.write(str(finetuningNet(train_data, train_batchSize, layerNum)))

    with open(finetune_test_proto, 'w') as f1:
        f1.write(str(finetuningNet(test_data, test_batchSize, layerNum)))

    paraConfig['snapshot_prefix'] = snapshot_prefix + '_finetune'
    finetune_solver = solver(finetune_train_proto, finetune_test_proto, paraConfig, save_path=finetune_solver_proto)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    finetune_net = caffe.SGDSolver(finetune_solver)

    for i in range(layerNum):
        weights = autoSaveFolder + snapshot_prefix + str(i+1) + '.caffemodel'
        finetune_net.net.copy_from(weights)

    niter = paraConfig['max_iter']
    test_interval = paraConfig['test_interval']
    # losses will also be stored in the log
    train_loss = np.zeros((niter,))
    test_loss = np.zeros((niter // test_interval,))

    isTest = paraConfig['isTest']
    # the main solver loop
    for it in range(niter):
        finetune_net.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = finetune_net.net.blobs['loss'].data

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if isTest:
            if it % test_interval == 0:
                print 'Iteration', it, 'testing...'
                loss_test = 0
                for test_it in range(paraConfig['test_iter']):
                    finetune_net.test_nets[0].forward()
                    loss_test += np.sum(finetune_net.test_nets[0].blobs['loss'].data)

                test_loss[it // test_interval] = loss_test / (paraConfig['test_iter'] * paraConfig['test_batchSize'])

    finetune_net.net.save(autoSaveFolder + snapshot_prefix + '_final' + '.caffemodel')
    print autoSaveFolder + snapshot_prefix + '_final' + '.caffemodel' + ' saved'
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(niter), train_loss / paraConfig['train_batchSize'])
    ax2.plot(test_interval * np.arange(int(np.ceil(niter / test_interval))),
             test_loss[0:int(np.ceil(niter / test_interval))], 'r')

    plt.title('finetune')
    plt.draw()

    print test_loss[0], test_loss[int(np.ceil(niter / test_interval)) - 1]
    print train_loss[0] / 256, train_loss[-1] / 256

    return 0

def classification_train(paraConfig, classification_train_proto= '../../Data/autoEncoder/classification_train.prototxt',
                         classification_test_proto='../../Data/autoEncoder/classification_test.prototxt',
                         classification_solver_proto='../../Data/autoEncoder/classification_solver.prototxt'):
    train_data = paraConfig['train_data']
    test_data = paraConfig['test_data']
    autoSaveFolder = paraConfig['autoSaveFolder']
    layerNeuronNum = paraConfig['layerNeuronNum']
    layerNum = len(layerNeuronNum) - 1
    train_batchSize = paraConfig['train_batchSize']
    test_batchSize = paraConfig['test_batchSize']
    snapshot_prefix = paraConfig['snapshot_prefix']
    learned_param = paraConfig['learned_param']

    with open(classification_train_proto, 'w') as f1:
        f1.write(str(classificationNet(train_data, train_batchSize, layerNeuronNum, layerNum, 4, learned_param)))

    with open(classification_test_proto, 'w') as f1:
        f1.write(str(classificationNet(test_data, test_batchSize, layerNeuronNum, layerNum, 4, learned_param)))

    paraConfig['snapshot_prefix'] = snapshot_prefix + '_classification'
    classification_solver = solver(classification_train_proto, classification_test_proto, paraConfig, save_path=classification_solver_proto)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    classification_net = caffe.SGDSolver(classification_solver)

    weights = autoSaveFolder + snapshot_prefix + '_final.caffemodel'
    classification_net.net.copy_from(weights)

    niter = paraConfig['max_iter']
    test_interval = paraConfig['test_interval']
    # losses will also be stored in the log
    train_loss = np.zeros((niter,))
    test_loss = np.zeros((niter // test_interval,))

    isTest = paraConfig['isTest']
    # the main solver loop
    for it in range(niter):
        classification_net.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = classification_net.net.blobs['loss'].data

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if isTest:
            if it % test_interval == 0:
                print 'Iteration', it, 'testing...'
                loss_test = 0
                for test_it in range(paraConfig['test_iter']):
                    classification_net.test_nets[0].forward()
                    loss_test += np.sum(classification_net.test_nets[0].blobs['loss'].data)

                test_loss[it // test_interval] = loss_test / (paraConfig['test_iter'] * paraConfig['test_batchSize'])

    classification_net.net.save(autoSaveFolder + snapshot_prefix + '_classification_final' + '.caffemodel')
    print autoSaveFolder + snapshot_prefix + '_classification_final.caffemodel' + ' saved'
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(niter), train_loss / paraConfig['train_batchSize'])
    ax2.plot(test_interval * np.arange(int(np.ceil(niter / test_interval))),
             test_loss[0:int(np.ceil(niter / test_interval))], 'r')

    plt.title('classification')
    plt.draw()

    print test_loss[0], test_loss[int(np.ceil(niter / test_interval)) - 1]
    print train_loss[0] / 256, train_loss[-1] / 256

    return 0

if __name__ == '__main__':
    # patchDataPath = '../../Data/one_in_minute_patch_test_diff_mean.hdf5'
    patchDataPath = '../../Data/type4_test_same_mean_s28_special.hdf5'
    # patchData_mean = '../../Data/patchData_mean.txt'
    # fm = open(patchData_mean, 'r')
    # mean_value = float(fm.readline().split(' ')[1])
    f = h5py.File(patchDataPath, 'r')
    data = f.get('data')
    test_num = data.shape[0]
    # d = np.array(data[0:10, :, :, :])
    # data = f['data']
    # print data.shape, d.shape
    for n in f:
        print n

    f.close()

    paraConfig = {}
    paraConfig['train_batchSize'] = 256
    paraConfig['test_batchSize'] = 278
    paraConfig['test_interval'] = 10000
    paraConfig['test_iter'] = test_num / paraConfig['test_batchSize']
    # paraConfig['test_iter'] = 100
    paraConfig['type'] = 'SGD'
    paraConfig['base_lr'] = 0.01
    paraConfig['lr_policy'] = 'step'
    paraConfig['gamma'] = 0.1
    paraConfig['stepsize'] = 2000
    paraConfig['momentum'] = 0.9
    paraConfig['weight_decay'] = 0.0
    paraConfig['max_iter'] = 100000
    paraConfig['display'] = 1000
    paraConfig['snapshot'] = 50000
    paraConfig['snapshot_prefix'] = 'layer_same_mean_s28_special'
    paraConfig['solver_mode'] = caffe_pb2.SolverParameter.GPU
    # paraConfig['train_data'] = '../../Data/type4_train_same_mean_s28_special.txt'
    # paraConfig['test_data'] = '../../Data/type4_test_same_mean_s28_special.txt'

    # ---balance for classification---
    paraConfig['train_data'] = '../../Data/type4_same_mean_s28_special_train.txt'
    paraConfig['test_data'] = '../../Data/type4_same_mean_s28_special_test.txt'
    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param = dict(lr_mult=2, decay_mult=0)
    learned_param = [weight_param, bias_param]
    frozen_param = [dict(lr_mult=0)] * 2
    layerNeuronNum = [28 * 28, 1000, 1000, 500, 64]
    drop_ratio = 0.2

    paraConfig['learned_param'] = learned_param
    paraConfig['frozen_param'] = frozen_param
    paraConfig['layerNeuronNum'] = layerNeuronNum
    paraConfig['drop_ratio'] = drop_ratio
    paraConfig['autoSaveFolder'] = '../../Data/autoEncoder/'
    paraConfig['layer_idx'] = ''
    paraConfig['isTest'] = False

    # with open('../../Data/autoEncoder/train_test.prototxt', 'w') as f1:
    #     f1.write(str(layerwise_train_net(3, paraConfig, True)))

    # with open('../../Data/autoEncoder/finetuning_train.prototxt', 'w') as f1:
    #     f1.write(str(finetuningNet('/home/ljm/NiuChuang/KLSA-auroral-images/Data/patchDataTrain.txt', 64, 4)))

    # with open('../../Data/autoEncoder/auto_encoder_test.prototxt', 'w') as f1:
    #     f1.write(str(layerwise_train_net('/home/ljm/NiuChuang/KLSA-auroral-images/Data/patchDataTest.txt', 100, 0)))
    #
    # auto_encoder_solver = solver('../../Data/autoEncoder/auto_encoder_train.prototxt',
    #                              '../../DataEncoder/auto_encoder_test.prototxt', paraConfig,
    #                              save_path='../../Data/autoEncoder/solver_test.prototxt')

    # with open('../../Data/autoEncoder/train_test.prototxt', 'w') as f1:
    #     f1.write(str(defineTestNet((10000,1,28,28), 4)))
    # f1.close()

    # --------------layer wise train, pretraining-----------
    # layerwise_train(paraConfig)
    # --------------finetune--------------------------------
    # finetue_train(paraConfig)
    #---------------classification--------------------------
    classification_train(paraConfig)

    plt.show()