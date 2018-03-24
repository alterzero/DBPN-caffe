function [result,t_run] = run_cnn(im,model,weights,use_gpu)
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% Weights (parameter) file
net_model = model;
net_weights = weights;
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download Model before you run this demo');
end

%%%%% change PROTOTXT DEPLOY  %%%%%
[wid,hei,channels_num] = size(im);
fidin1=fopen(net_model,'r+');
i=0;
while ~feof(fidin1)
    tline=fgetl(fidin1);
    i=i+1;
    newtline{i}=tline;
    if i == 4
        newtline{i}=[tline(1:11) num2str(channels_num)];
    end
    if i == 5
        newtline{i}=[tline(1:11) num2str(hei)];
    end
    if i == 6
        newtline{i}=[tline(1:11) num2str(wid)];
    end
end
fclose(fidin1);
fidin1=fopen(net_model,'w+');
for j=1:i
    fprintf(fidin1,'%s\n',newtline{j});
end
fclose(fidin1);
%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

%run network
tic;
out = net.forward({im});
t_run=toc;
result=out{1};
%end

%result=blockproc( reshape(1:dd,a/aa,b/bb)', [1,1], @(x) im_out(:,:,x.data) );
% call caffe.reset_all() to reset caffe
caffe.reset_all();

