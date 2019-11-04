function net = cnn_xor_simple(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts.layers_num = 1;
opts.final_fc = 1;
opts.nfeat = 10;
opts.imdbEval = 0;
opts.non_linearity = 'relu';
opts.pool = 'max';
opts = vl_argparse(opts, varargin) ;
opts.nfeat = 2;

rng('default');
rng(0) ;
num_feat_hidden=2
f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,opts.nfeat,num_feat_hidden, 'single'), zeros(1, num_feat_hidden, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;


                       
% for lx = 1:opts.layers_num
%     net.layers{end+1} = struct('type', 'conv', ...
%                                'weights', {{f*randn(1,1,opts.nfeat,opts.nfeat, 'single'),zeros(1,opts.nfeat,'single')}}, ...
%                                'stride', 1, ...
%                                'pad', 0) ;                     
%     if strcmp(opts.non_linearity, 'relu1')                       
%         net.layers{end+1} = struct('type', 'relu', 'leak', 0.1) ;
%     else
%         net.layers{end+1} = struct('type', opts.non_linearity) ;
%     end 
% 
% 
% end
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,num_feat_hidden,num_feat_hidden, 'single'),zeros(1,num_feat_hidden,'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;                     
% net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,num_feat_hidden,2, 'single'), zeros(1,2,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

if  opts.imdbEval == 0
    net.layers{end+1} = struct('type', 'softmaxloss') ;
else
    net.layers{end+1} = struct('type', 'softmax') ;
end
% optionally switch to batch normalization
%if opts.batchNormalization
%  net = insertBnorm(net, 1) ;
%  net = insertBnorm(net, 4) ;
%  net = insertBnorm(net, 7) ;
%end

% Meta parameters
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 15 ;
net.meta.trainOpts.batchSize = 100 ;


% Fill in defaul values
net = vl_simplenn_tidy(net) ;

vl_simplenn_display(net)

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {'prediction', 'label'}, 'error') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
      'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.weights{2} = [] ;  % eliminate bias in previous conv layer
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
