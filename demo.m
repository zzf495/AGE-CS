%% Implementation of AGE-CS
%%% Paper: Adaptive Graph Embedding with Consistency and Specificity for Domain Adaptation

clc; clear all;
addpath(genpath('./util/'));
srcStr = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};
finalResult=[];

%% initialize the Ytpseudo by a basic classifier or not 
options.init=1; % if `options.init==1`, then Ytpseudo is assigned before training
options.classify=2; % `options.classify==1`, use KNN
                    % Otherwise, use SRM
options.gamma=1; % The parameter of SRM
options.Kernel=2; % The parameter of SRM
options.mu=0.1; % The parameter of SRM
%% The hyper-parameters of AGE-CS
options.T=10; % iteration
options.dim=50; % dimension
options.k=32; % neighborhood number
options.aug=0.1; % smooth parameter
options.M_mu=0.3; % the weight between marginal and conditional distributions
options.alpha=5; % the weight of Laplace matrix (manifold regularization)
options.lambda=0.1; % the weight of regularization for projection matrix
options.tau=1e-3; % the weight to emphasize the semantic information
%% Run the experiments
for i = 1:12
    src = char(srcStr{i});
    tgt = char(tgtStr{i});
    fprintf('%d: %s_vs_%s\n',i,src,tgt);
    load(['./data/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs =zscore(fts,1);
    Ys = labels;
    load(['./data/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xt = zscore(fts,1);
    Yt = labels;
    Xs=Xs';
    Xt=Xt';
    [~,result,~] = AGE_CS(Xs,Ys,Xt,Yt,options);
    finalResult=[finalResult;result];
end

