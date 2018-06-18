sfx = 'simplenn';
for final_fc = 0:1
for layers_num = 0:20
dst_file = ['..\..\data\features\mnist-baseline-' num2str(0) sfx '_jitter_' num2str(1) '_layers_num_' num2str(layers_num) '_final_fc_' num2str(final_fc) '_features.mat'];
src_file = ['..\..\data\mnist-baseline-' num2str(0) sfx '_jitter_' num2str(1) '_layers_num_' num2str(layers_num) '_final_fc_' num2str(final_fc) '\features_0.mat'];
system(['copy ' src_file ' ' dst_file]);
end 
end