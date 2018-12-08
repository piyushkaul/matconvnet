%% Experiment with the cnn_mnist_fc_bnorm
addpath('..\..\..\diffgeom');

for final_fc = 1
for layers_num = 5
    [net_bn, info_bn] = cnn_mnist('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 0);

    epoch_dir = getPath('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 0);   
    eval_dir = getPath('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 1);

    system(['mkdir -p ' eval_dir])
    system(['copy ' epoch_dir '\net-epoch-20.mat' ' ' eval_dir '\net-epoch-19.mat']);   

    [net_bn, info_bn] = cnn_mnist('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 1);
    
    diffgeom_minst('features_0.mat')
    
end
end 


