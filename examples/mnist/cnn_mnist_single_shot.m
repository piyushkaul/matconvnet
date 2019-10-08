%% Experiment with the cnn_mnist_fc_bnorm
addpath('..\..\..\diffgeom');
non_linearity  = 'avg';
for final_fc = 1
for layers_num = 5
    [net_bn, info_bn] = cnn_mnist('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 0, 'non_linearity', non_linearity);

    epoch_dir = getPath('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 0, 'non_linearity', non_linearity);   
    %eval_dir = getPath('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 1, 'non_linearity', non_linearity);

    %system(['mkdir -p ' eval_dir])
    %system(['copy ' epoch_dir '\net-epoch-20.mat' ' ' eval_dir '\net-epoch-19.mat']);   
    system(['copy ' fullfile(vl_rootnn, 'data', 'eval', 'imdb_eval.mat') ' ' epoch_dir])

    [net_bn, info_bn] = cnn_mnist('batchNormalization', false, 'layers_num', layers_num, 'final_fc', final_fc, 'imdbEval', 1, 'non_linearity', non_linearity);
    
    diffgeom_minst('features_0.mat')
    
end
end 


