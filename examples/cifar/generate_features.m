for numLayers=10
cnn_cifar_riem2('layers',numLayers);
dos('del ..\..\data\net-epoch-127.mat')
dos('copy ..\..\data\cifar-lenet\features_1.mat ..\..\data\features_cifar\')
end