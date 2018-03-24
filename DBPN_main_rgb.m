close all;
clear all;

%% set parameters
use_gpu=0;
up_scale = 8;
ensemble=0;
dataset = 'Set5/';
testfolder = ['Test/' dataset];
resultfolder=['Result/' int2str(up_scale) 'x/' dataset];

%% flat or deconv
flat=0;
model = 'model/DBPN_mat_8x.prototxt';
weights = 'model/DBPN_8x.caffemodel';

if ~exist(resultfolder,'file')
    mkdir(resultfolder);
end

filepaths = dir(fullfile(testfolder,'*.png'));
[aa,bb]=size(filepaths);
if aa==0
    filepaths = dir(fullfile(testfolder,'*.bmp'));
end


tt_bic = zeros(length(filepaths),1);
tt_dbpn = zeros(length(filepaths),1);

psnr_bic = zeros(length(filepaths),1);
psnr_dbpn = zeros(length(filepaths),1);

ssim_bic = zeros(length(filepaths),1);
ssim_dbpn = zeros(length(filepaths),1);

fsim_bic = zeros(length(filepaths),1);
fsim_dbpn = zeros(length(filepaths),1);

for i = 1 : length(filepaths)
   
    %% read ground truth image
    [add,imname,type] = fileparts(filepaths(i).name);
    im = imread([testfolder imname type]);
    
    if size(im,3) == 1
        im=im(:,:,[1 1 1]);
    end
    
    %% work on illuminance only
    im_gnd = modcrop(im, up_scale);
    im_gnd = single(im_gnd)/255;
    im_l = imresize(im_gnd, 1/up_scale, 'bicubic');

    %% bicubic interpolation
    tic
    im_b = imresize(im_l, up_scale, 'bicubic');
    t_bic=toc;
   
    %% DBPN
    if flat==1
        [im_dbpn,t_dbpn] = run_cnn(im_b, model, weights,use_gpu);
        t_dbpn=t_dbpn+t_bic;
    else
        if ensemble==0
            [im_dbpn,t_dbpn] = run_cnn(im_l, model, weights,use_gpu);
        else
            [im_dbpn,t_dbpn]=selfEnsemble(im_l,model,weights,use_gpu);
        end
    end

    %% remove border
     im_dbpn = shave(uint8(im_dbpn * 255), [up_scale, up_scale]);
     im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
     im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);
    
    %% compute time
    tt_bic(i) = t_bic;
    tt_dbpn(i) = t_dbpn+t_bic;

    %% compute PSNR
    psnr_bic(i) = compute_psnr(im_gnd,im_b);
    psnr_dbpn(i) = compute_psnr(im_gnd,im_dbpn);
    
    %% compute FSIM
    fsim_bic(i) = FeatureSIM(im_gnd,im_b);
    fsim_dbpn(i) = FeatureSIM(im_gnd,im_dbpn);
     
     %% compute SSIM
    ssim_bic(i) = ssim_index(im_gnd,im_b);
    ssim_dbpn(i) = ssim_index(im_gnd,im_dbpn);

    %% save results
    %imwrite(im_b, [resultfolder imname '_bic.bmp']);
    %imwrite(im_dbpn, [resultfolder imname '_dbpn.bmp']);

end

fprintf('Bicubic: %f , %f , %f , %f \n', mean(psnr_bic), mean(ssim_bic), mean(fsim_bic), mean(tt_bic));
fprintf('dbpn: %f , %f , %f , %f \n', mean(psnr_dbpn), mean(ssim_dbpn), mean(fsim_dbpn), mean(tt_dbpn));
