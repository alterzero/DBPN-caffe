function [im_drl,t_drbpsr]=selfEnsemble(im_l,model,weights,use_gpu)
        t_drbpsr=0;
        im_l_90=imrotate(im_l,90);
        im_l_180=imrotate(im_l,180);
        im_l_270=imrotate(im_l,270);
        [im_drl_0,t_drbpsr_0] = run_cnn(im_l, model, weights,use_gpu);
        [im_drl_90,t_drbpsr_90] = run_cnn(im_l_90, model, weights,use_gpu);
        [im_drl_180,t_drbpsr_180] = run_cnn(im_l_180, model, weights,use_gpu);
        [im_drl_270,t_drbpsr_270] = run_cnn(im_l_270, model, weights,use_gpu);
        [cc,vv,channel]=size(im_drl_0);
        im_drl=zeros(cc,vv,channel,8);
        
        t_drbpsr=t_drbpsr+t_drbpsr_0+t_drbpsr_90+t_drbpsr_180+t_drbpsr_270;
        im_drl(:,:,:,1)=im_drl_0;
        im_drl(:,:,:,2)=imrotate(im_drl_90,-90);
        im_drl(:,:,:,3)=imrotate(im_drl_180,-180);
        im_drl(:,:,:,4)=imrotate(im_drl_270,-270);
        
        im_l=flip(im_l,2);
        im_l_90=imrotate(im_l,90);
        im_l_180=imrotate(im_l,180);
        im_l_270=imrotate(im_l,270);
        [im_drl_0,t_drbpsr_0] = run_cnn(im_l, model, weights,use_gpu);
        [im_drl_90,t_drbpsr_90] = run_cnn(im_l_90, model, weights,use_gpu);
        [im_drl_180,t_drbpsr_180] = run_cnn(im_l_180, model, weights,use_gpu);
        [im_drl_270,t_drbpsr_270] = run_cnn(im_l_270, model, weights,use_gpu);
        
        t_drbpsr=t_drbpsr+t_drbpsr_0+t_drbpsr_90+t_drbpsr_180+t_drbpsr_270;
        im_drl(:,:,:,5)=flip(im_drl_0,2);
        im_drl(:,:,:,6)=flip(imrotate(im_drl_90,-90),2);
        im_drl(:,:,:,7)=flip(imrotate(im_drl_180,-180),2);
        im_drl(:,:,:,8)=flip(imrotate(im_drl_270,-270),2);
        
        im_drl=mean(im_drl,4);