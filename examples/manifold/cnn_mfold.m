function [net, info] = cnn_mfold(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.mnistVer = 0;
opts.imdbEval = 0;
opts.dataJitter = 1;

opts.ntrain=10000;
opts.ntest=1000;
opts.nfeat=1000;

opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.layers_num = 10;
opts.final_fc = 1;
opts.non_linearity = 'relu';
opts.pool = 'max';
opts.layers_clip=14;
opts.finalSum=1;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['mfold-baseline-10-' num2str(opts.mnistVer) sfx '_jitter_' num2str(opts.dataJitter) '_layers_num_' num2str(opts.layers_num) '_final_fc_' num2str(opts.final_fc) '_nl_' opts.non_linearity]) ;


[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.imdbPathEval = fullfile(opts.expDir, 'imdb_eval.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
   
   net = cnn_mfold_generic('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType, 'layers_num', opts.layers_num, 'final_fc', opts.final_fc, 'nfeat', opts.nfeat, 'non_linearity', opts.non_linearity, 'pool', opts.pool) ;
           

else
  net = opts.network ;
  opts.network = [] ;
end

if opts.imdbEval == 0
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMfoldImdb(opts) ;
  mkdir(opts.expDir) ;
  fprintf(['creating file ' opts.expDir])
  save(opts.imdbPath, '-struct', 'imdb') ;
end
else
if exist(opts.imdbPathEval, 'file')
  imdb = load(opts.imdbPathEval) ;
else
  imdb = getMfoldImdbEval(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPathEval, '-struct', 'imdb') ;
end
end 


net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3),...
  'mnistVer', opts.mnistVer,...
 'imdbEval', opts.imdbEval,...
  'layers_clip',opts.layers_clip,...
  'finalSum',opts.finalSum);

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMfoldImdb(opts)
% --------------------------------------------------------------------

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end
rng(0) 
%x1=rand([1,1,opts.nfeat,1]) ;
%x2=rand([1,1,opts.nfeat,2]) ;
u1 = rand([opts.nfeat,1]);
u2 = rand([opts.nfeat,1]);
u1 = u1./norm(u1(:));
u2 = u2./norm(u2(:));

theta = rand([1,opts.ntrain+opts.ntest])*2*pi;
classes = randi(2,1,opts.ntrain+opts.ntest);
idx1 = find(classes==1);
idx2 = find(classes==2);
delta_theta = zeros(size(theta));
delta_theta(idx2) = pi;

x_t = theta.*(cos(theta+delta_theta) ) + 0.5.*randn(size(theta));
y_t = theta.*(sin(theta+delta_theta) ) + 0.5.*randn(size(theta));
figure;plot(x_t(idx1),y_t(idx1),'r.');hold on; plot(x_t(idx2),y_t(idx2),'b.');

z_t = u1 * x_t +  u2 * y_t;

data = single(shiftdim(z_t,-2));

%y1=ones(1,ntrain);
%y2=2*ones(1,ntest);

set = [ones(1,opts.ntrain) 3*ones(1,opts.ntest)];


imdb.images.data = data ;
imdb.images.labels = classes;%cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

% --------------------------------------------------------------------
function imdb = getMfoldImdbEval(opts)
% --------------------------------------------------------------------
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

rng(0) 
u1 = rand([opts.nfeat,1]);
u2 = rand([opts.nfeat,1]);
u1 = u1./norm(u1(:));
u2 = u2./norm(u2(:));

min_x = -8;
max_x = 8;
min_y = -8;
max_y = 8;

steps_per_x = 0.05;
steps_per_y = 0.05;
x_range = min_x:steps_per_x:max_x;
y_range = min_y:steps_per_y:max_y;

outdata = zeros(1,1,opts.nfeat,numel(x_range)*numel(y_range));

theta_used = [];
tx_used = [];
vec=rand([1,1,opts.nfeat,1]);
itx = 0;
x_t_total = [];
y_t_total = [];

for x_t = x_range    
    for y_t = y_range
        itx = itx + 1;

        x_t2 = x_t ;%+ 0.2.*randn(size(u2));
        y_t2 = y_t ;%+ 0.2.*randn(size(u2));
        z_t = x_t2 .* u1 + y_t2 .* u2;

        x_t_total = [x_t_total x_t2];
        y_t_total = [y_t_total y_t2];

        outvec = single(shiftdim(z_t,-2));      
        outdata(1,1,:,itx) = outvec;

        tx_used(itx) = x_t2;
        theta_used(itx) = y_t2;    

        if mod(itx,100) == 0
            fprintf('.')
        end 
        
        if mod(itx,5000) == 0
            fprintf('\n')
        end     
        
    end 
end 
figure;plot(x_t_total, y_t_total, 'r.')
outdata=cat(4,outdata,zeros(1,1,opts.nfeat,1));
imdb.images.data = single(outdata);
sz = size(outdata);
imdb.images.labels = 1*ones(1,sz(4),'single');
imdb.images.set = 3*ones(1,sz(4),'single');
imdb.images.set(end) = 1;
imdb.images.tx_used = tx_used;
imdb.images.theta_used = theta_used;

% --------------------------------------------------------------------
function imdb = getMfoldImdbEval2(opts)
% --------------------------------------------------------------------
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end
rng(0) 
u1 = rand([opts.nfeat,1]);
u2 = rand([opts.nfeat,1]);
u1 = u1./norm(u1(:));
u2 = u2./norm(u2(:));

min_deg = 0;
max_deg=2*pi;
steps_per_deg=2*pi/360;
max_tx = pi;
min_tx = 0;
steps_per_tx = pi/180;
tx_range = min_tx:steps_per_tx:max_tx;
theta_range = min_deg:steps_per_deg:max_deg;
outdata = zeros(1,1,opts.nfeat,numel(theta_range)*numel(tx_range));
theta_used = [];
tx_used = [];
vec=rand([1,1,opts.nfeat,1]);
itx = 0;
x_t_total = [];
y_t_total = [];

for theta = theta_range    
    for tx = tx_range
        itx = itx + 1;

        x_t = theta.*cos(theta + tx) + 0.2.*randn(size(u2));
        y_t = theta.*sin(theta + tx) + 0.2.*randn(size(u2));
        z_t = x_t .* u1 + y_t .* u2;

        x_t_total = [x_t_total x_t];
        y_t_total = [y_t_total y_t];

        outvec = single(shiftdim(z_t,-2));      
        outdata(1,1,:,itx) = outvec;

        tx_used(itx) = tx;
        theta_used(itx) = theta;    

        if mod(itx,100) == 0
            fprintf('.')
        end 
        if mod(itx,5000) == 0
            fprintf('\n')
        end     
    end 
end 
figure;plot(x_t_total, y_t_total, 'r.')
outdata=cat(4,outdata,zeros(1,1,opts.nfeat,1));
imdb.images.data = single(outdata);
sz = size(outdata);
imdb.images.labels = 1*ones(1,sz(4),'single');
imdb.images.set = 3*ones(1,sz(4),'single');
imdb.images.set(end) = 1;
imdb.images.tx_used = tx_used;
imdb.images.theta_used = theta_used;
