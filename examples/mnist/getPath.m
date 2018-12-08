function expDir = getPath(varargin)
opts.mnistVer = 0;
opts.imdbEval = 0;
opts.dataJitter = 0;
opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.layers_num = 1;
opts.final_fc = 1;
opts.first_digit = 6;
opts.second_digit = 9;
opts.mnistVer = 1;
[opts, varargin] = vl_argparse(opts, varargin) ;
expDir = fullfile(vl_rootnn, 'data', ['mnist-new2-' num2str(opts.imdbEval)  '_jitter_' num2str(opts.dataJitter) '_layers_num_' num2str(opts.layers_num) '_final_fc_' num2str(opts.final_fc) '_first_digit_' num2str(opts.first_digit)  '_second_digit_' num2str(opts.second_digit)]) ;
return ;