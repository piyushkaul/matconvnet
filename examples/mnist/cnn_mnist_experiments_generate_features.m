src_imdb = '..\..\data\mnist-baseline-5simplenn1\imdb_eval.mat';
sfx = 'simplenn';
for final_fc = 0:1
for layers_num = 0:20
src_file = ['..\..\data\final_epochs\mnist-baseline-' num2str(0) sfx '_jitter_' num2str(1) '_layers_num_' num2str(layers_num) '_final_fc_' num2str(final_fc) '_net-epoch-20.mat'];
destination_directory = ['..\..\data\mnist-baseline-' num2str(0) sfx '_jitter_' num2str(1) '_layers_num_' num2str(layers_num) '_final_fc_' num2str(final_fc) '\'];
system(['mkdir -p ' destination_directory])
system(['copy ' src_file ' ' destination_directory 'net-epoch-20.mat']);
system(['copy ' src_imdb ' ' destination_directory]);
[net_bn, info_bn] = cnn_mnist(...
   'batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc);
end
end 