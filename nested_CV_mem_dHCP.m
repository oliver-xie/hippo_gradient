% author: OX 12/18/23
% final model used for memory prediction 

clear 
load('pmap.mat');
nrep = 1;
dim = [.7 .1 .3 .3];
opts = statset('UseParallel',true);
Lambda = logspace(-1,0,100);
%% Right projection map
predicted_mem_R = zeros(length(memscore_valid),nrep);
for rep = 1 : nrep % repeat #nrep times
    CVO = cvpartition(round(memscore_valid/10),'KFold',10);
    for k_out = 1 : CVO.NumTestSets
        trIdx = find(CVO.training(k_out));
        teIdx = find(CVO.test(k_out));
        trainset = pmap_mean_R_(trIdx,:);
        testset = pmap_mean_R_(teIdx,:);
        mean_ = mean(trainset,1);
        std_ = std(trainset,1);
        trainset = zscore(trainset,[],1);
        testset = testset-repmat(mean_,[length(teIdx),1]);
        testset = testset./repmat(std_,[length(teIdx),1]);
        CVI = cvpartition(round(memscore_valid(trIdx)/10),'KFold',10); % inner-CV
        CVMdl = fitrlinear(trainset,memscore_valid(trIdx),'CVPartition',CVI,'Lambda',Lambda,...
            'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
        predicted = CVMdl.kfoldPredict;
        [innerloss(rep,k_out),minloss_idx(rep,k_out)] = min(CVMdl.kfoldLoss);
        [c_inner(rep,k_out),p_inner(rep,k_out)] = corr(predicted(:,minloss_idx(rep,k_out)),memscore_valid(trIdx),'type','spearman');
        beta = zeros(size(trainset,2),CVMdl.KFold);
        bias = zeros(1,CVMdl.KFold);
        for k_inner = 1 : CVMdl.KFold % avg 10-fold inner CV to get the model to test on outer loop
            tmp = CVMdl.Trained{k_inner}.Beta;
            beta(:,k_inner) = tmp(:,minloss_idx(rep,k_out));
            tmp = CVMdl.Trained{k_inner}.Bias;
            bias(k_inner) = tmp(minloss_idx(rep,k_out));
        end
        beta_final(:,k_out,rep) = mean(beta,2);
        bias_final(k_out,rep) = mean(bias);
        predicted_mem_R(teIdx,rep) = testset*squeeze(beta_final(:,k_out,rep)) + bias_final(k_out,rep);
    end
    [c_outer_R(rep),p_outer_R(rep)] = corr(predicted_mem_R(:,rep),memscore_valid,'type','spearman');
    outloss_R(rep) = mean((predicted_mem_R(:,rep)-memscore_valid).^2);
    figure
    scatter(predicted_mem_R(:,rep),memscore_valid)
    xlim([100,120])
    refline
    title('true vs predicted mem (R Hippo)')
    str = ['c = ' num2str(c_outer_R(rep),3) newline 'p = ' num2str(p_outer_R(rep),2)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on');
end
[c_outer_R_avg,p_outer_R_avg] = corr(mean(predicted_mem_R,2),memscore_valid);
[cp_outer_R_avg,pp_outer_R_avg] = partialcorr(mean(predicted_mem_R,2),memscore_valid,[mfd_valid age_valid]);
if nrep > 1
    figure
    scatter(mean(predicted_mem_R,2),memscore_valid)
    h = refline;
    h.LineWidth = 3;
    str = ['r = ' num2str(c_outer_R_avg,3) newline 'p = ' num2str(p_outer_R_avg,2)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on');
    title(['mem prediction true vs predicted  (R) corr\_p =' num2str(p_outer_R_avg,3)])
    xlabel('predicted')
    ylabel('true')
end
beta_final_p_r= mean(beta_final~=0,[2 3]);
%% Left projection map
predicted_mem_L = zeros(length(memscore_valid),nrep);
for rep = 1 : nrep % repeat nrep times
    CVO = cvpartition(round(memscore_valid/10),'KFold',10);
    for k_out = 1 : CVO.NumTestSets
        trIdx = find(CVO.training(k_out));
        teIdx = find(CVO.test(k_out));
        trainset = pmap_mean_L_(trIdx,:);
        testset = pmap_mean_L_(teIdx,:);
        mean_ = mean(trainset,1);
        std_ = std(trainset,1);
        trainset = zscore(trainset,[],1);
        testset = testset-repmat(mean_,[length(teIdx),1]);
        testset = testset./repmat(std_,[length(teIdx),1]);
        CVI = cvpartition(round(memscore_valid(trIdx)/10),'KFold',10); % inner-CV
        CVMdl = fitrlinear(trainset,memscore_valid(trIdx),'CVPartition',CVI,'Lambda',Lambda,...
            'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
        predicted = CVMdl.kfoldPredict;
        [innerloss(rep,k_out),minloss_idx(rep,k_out)] = min(CVMdl.kfoldLoss);
        [c_inner(rep,k_out),p_inner(rep,k_out)] = corr(predicted(:,minloss_idx(rep,k_out)),memscore_valid(trIdx),'type','spearman');
        beta = zeros(size(trainset,2),CVMdl.KFold);
        bias = zeros(1,CVMdl.KFold);
        for k_inner = 1 : CVMdl.KFold % avg folds to get the model
            tmp = CVMdl.Trained{k_inner}.Beta;
            beta(:,k_inner) = tmp(:,minloss_idx(rep,k_out));
            tmp = CVMdl.Trained{k_inner}.Bias;
            bias(k_inner) = tmp(minloss_idx(rep,k_out));
        end
        beta_final(:,k_out,rep) = mean(beta,2);
        bias_final(k_out,rep) = mean(bias);
        predicted_mem_L(teIdx,rep) = testset*squeeze(beta_final(:,k_out,rep)) + bias_final(k_out,rep);
    end
    [c_outer_L(rep),p_outer_L(rep)] = corr(predicted_mem_L(:,rep),memscore_valid,'type','spearman');
    outloss_L(rep) = mean((predicted_mem_L(:,rep)-memscore_valid).^2);
    figure
    scatter(predicted_mem_L(:,rep),memscore_valid)
    xlim([100,120])
    refline
    title('true vs predicted mem (L Hippo)')
    str = ['c = ' num2str(c_outer_L(rep),3) newline 'p = ' num2str(p_outer_L(rep),2)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on');
end

beta_final_p_L= mean(beta_final~=0,[2 3]);
if nrep > 1
figure
scatter(mean(predicted_mem_L,2),memscore_valid)
[c_outer_L_avg,p_outer_L_avg] = corr(mean(predicted_mem_L,2),memscore_valid);
[cp_outer_L_avg,pp_outer_L_avg] = partialcorr(mean(predicted_mem_L,2),memscore_valid,[mfd_valid age_valid]);
h = refline;
h.LineWidth = 3;
str = ['r = ' num2str(c_outer_L_avg,3) newline 'p = ' num2str(p_outer_L_avg,2)];
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title(['mem prediction true vs predicted (L) p =' num2str(p_outer_L_avg,3)])
xlabel('predicted')
ylabel('true')
end

