epoch_store='C:\GitHub\matconvnet\data\mfold-baseline-5-0simplenn-bnorm_jitter_1_layers_num_2_final_fc_1_nl_relu';
features_database='C:\GitHub\matconvnet\data\features_all_epochs_relu';
system(['mkdir -p ' features_database])
for epoch = 1:15
src_file = [epoch_store '\net-epoch-' num2str(epoch) '.mat'];
destination_directory = ['..\..\data\mfold-baseline-trial-relu\'];
system(['mkdir -p ' destination_directory])
system(['copy ' src_file ' ' destination_directory 'net-epoch-15.mat']);
[net_bn, info_bn] = cnn_mfold('expDir',destination_directory, 'imdbEval', 1);
system(['del ' destination_directory 'net-epoch-16.mat'])
system(['copy ' destination_directory 'features_0.mat '  features_database '\features_' num2str(epoch) '.mat']);
end
