epoch_store='C:\GitHub\matconvnet\data\epoch_database2';
features_database='C:\GitHub\matconvnet\data\features_all_epochs2';

for epoch = 15
src_file = [epoch_store '\net-epoch-' num2str(epoch) '.mat'];
destination_directory = ['..\..\data\mfold-baseline-trial2\'];
system(['mkdir -p ' destination_directory])
system(['copy ' src_file ' ' destination_directory 'net-epoch-19.mat']);
[net_bn, info_bn] = cnn_mnist('expDir',destination_directory);
system(['del ' destination_directory 'net-epoch-20.mat'])
system(['copy ' destination_directory 'features_0.mat '  features_database '\features_' num2str(epoch) '.mat']);
end
