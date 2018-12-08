function y = vl_nnslicingrelu(x,varargin)
%VL_NNRELU CNN rectified linear unit.
%   Y = VL_NNRELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNRELU(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   VL_NNRELU(...,'OPT',VALUE,...) takes the following options:
%
%   `Leak`:: 0
%      Set the leak factor, a non-negative number. Y is equal to X if
%      X is not smaller than zero; otherwise, Y is equal to X
%      multipied by the leak factor. By default, the leak factor is
%      zero; for values greater than that one obtains the leaky ReLU
%      unit.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNRELU(X), then
%   VL_NNRELU(X,DZDY) gives the same result as VL_NNRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
  dzdy = varargin{1} ;
  varargin(1) = [] ;
else
  dzdy = [] ;
end

opts.leak = 0 ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;
width = size(x,1);
height = size(x,2);
depth = size(x,3);
batch = size(x,4);

for j=1:batch
    for i=1:depth
        w = gather(x(:,:,i,j));
        F(j,:,i) = w(:);
    end 
end
sz = size(x);
for j=1:batch
  FX = squeeze(F(j,:,:));
  [U,S,V]=svd(FX','econ');
  margin(:, j) = U(:,1)'*FX';
end 
if sz(1) ~= 1 || sz(2) ~= 1
margin = reshape(margin, [sz(1) sz(2) 1 sz(4)]);
z = repmat(margin, [1 1 sz(3) 1]);
else
    z = margin;
end 
idxes = find(z > 0);
if nargin <= 1 || isempty(dzdy)
    %y = max(x, 0) ;
    y = x;
    y(idxes) = 0;
else
    y = dzdy;
    y(idxes) = 0;
    %y = dzdy .* (x > 0) ;
end
