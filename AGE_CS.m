function [acc,acc_ite,obj] = AGE_CS(Xs,Ys,Xt,Yt,options)
    %% set parameters if NULL
    options=defaultOptions(options,'T',10,...
        'dim',100,...
        'aug',0.1,...
        'lambda',0.1,...
        'k',10,...
        'M_mu',0.1,...
        'tau',1e-3);
    %% init parameters
    dim=options.dim;% dimension
    aug=options.aug; % smooth parameter
    lambda=options.lambda; % the weight of regularization for projection matrix
    k=options.k;% neighborhood number
    mu=options.M_mu;% the weight between marginal and conditional distributions
    alpha=options.alpha; % the weight of Laplace matrix (manifold regularization)
    tau=options.tau; % the weight to emphasize the semantic information
    % -------------------------------
    Xs=normr(Xs')';
    Xt=normr(Xt')';
    obj=[];
    acc=0;
    acc_ite=[];
    
    X=[Xs,Xt];
    X=L2Norm(X')';
    [m,ns]=size(Xs);
    nt=size(Xt,2);
    n=ns+nt;
    C=length(unique(Ys));
    %% The neighborhood number should less than 'min(ns,nt)-1'
    if k>ns || k >nt
       fprintf('Replace k [%2d] as [%2d]\n',k,min(ns,nt)-1);
       k=min(ns,nt)-1; 
    end
    %% Decide whether initialize Ytpseudo
    Ytpseudo=[];
    if options.init==1
        fprintf('init pseudo-labels for target domain\n');
        if options.classify==1
            fprintf('Use KNN.\n');
            [Ytpseudo] = classifyKNN(Xs,Ys,Xt,1);
        else
           fprintf('Use SRM.\n');
           opt=options;
           [Ytpseudo,~,~] = SRM(Xs,Xt,Ys,opt);
        end
    else
         fprintf('init pseudo-labels for target domain as NULL\n');
    end
    %% init parameters for Ss
    clear opt;
    graphYs=hotmatrix(Ys,C,0);
    graphYs=graphYs*graphYs';
    opt.dist=my_dist(Xs');
    opt.semanticGraph=graphYs;
    opt.tau=tau;
    opt.k=k;
    [Ss_ori,rs]=AGE(opt); % ns * C
    
    %% init parameters for St
    if ~isempty(Ytpseudo)
        clear opt;
        opt.dist=my_dist(Xt');
        opt.tau=tau;
        opt.k=k;
        graphYtpseudo=hotmatrix(Ytpseudo,C)*hotmatrix(Ytpseudo,C)';
        opt.semanticGraph=graphYtpseudo;
        [St_ori,rt]=AGE(opt); % nt * C
    else
        St_ori=zeros(nt,nt);
        rt=0;
    end
    clear opt;
    M0=marginalDistribution(Xs,Xt,C);% the marginal distribution
    Ss=Ss_ori;
    St=St_ori;
    H=centeringMatrix(n);% the centering matrix
    right=X*H*X';% pre-calculation to save time
    fprintf('rs:%.4f, rt:%.4f\n',rs,rt);
    for i=1:options.T
        % calculate the whole S
        S=blkdiag(Ss,St);
        S=S+S';
        L = computeL_byW(S);
        if ~isempty(Ytpseudo)
            % marginal + conditional distributions
            M=(1-mu)*M0+mu*conditionalDistribution(Xs,Xt,Ys,Ytpseudo,C);
        else
            % marginal distribution
            M=M0;
        end
        L=L./norm(L,'fro');
        M=M./norm(M,'fro');
        %% update W by (28)
        left=X*(M+alpha*L)*X'+lambda*eye(m);
        [A,~]=eigs(left,right,dim,'sm');
        AX=A'*X;
        AX=L2Norm(AX')';
        AXs=AX(:,1:ns);
        AXt=AX(:,ns+1:end);
        if options.classify==1
            [Ytpseudo] = classifyKNN(AXs,Ys,AXt,1);
        else
           opt=options;
           [Ytpseudo,~,~] = SRM(AXs,AXt,Ys,options);
        end
        acc=getAcc(Ytpseudo,Yt);
        acc_ite(i)=acc;
        %% update Ss by Eq. (25)
        clear opt;
        opt.dist=my_dist(AXs');
        opt.semanticGraph=graphYs;
        opt.tau=tau;
        opt.k=k;
        opt.rr=rs; %gamma_s;
        [Ss_iter,rs]=AGE(opt);
        %% update St by Eq. (26)
        clear opt;
        opt.dist=my_dist(AXt');
        opt.k=k;
        %% update \hat{G}_t by Eq.(20)
        graphYtpseudo=hotmatrix(Ytpseudo,C)*hotmatrix(Ytpseudo,C)'; 
        opt.semanticGraph=graphYtpseudo;
        if options.init==0&& i==1
            % do nothing
        else
            opt.rr=rt;%gamma_t
        end
        opt.tau=tau;
        [St_iter,rt]=AGE(opt);
        %% Smooth S by Eq. (27)
        Ss=(1-aug)*Ss+aug*Ss_iter;
        %%% If Ytpseudo is empty, we set St(0)=St(1)
        if options.init==0&& i==1
            St=St_iter;
        else
            St=(1-aug)*St+aug*St_iter;
        end
        %% print the results
        fprintf('[%2d] acc:%.4f\n',i,acc*100);
%         %% calculate the objective function
%         obj(i)=trace(A'*left*A);
    end
end
function [L] = computeL_byW(W)
    n=size(W,1);
    Dw = diag(sparse(sqrt(1 ./ (sum(W)+eps) )));
    L = eye(n) - Dw * W * Dw;
end
function D = my_dist(fea_a,fea_b)
%% input:
%%% fea_a: n1*m
%%% fea_b: n2*m
    if nargin==1
        fea_b=0;
    end
    if nargin<=2
       [n1,n2]=size(fea_b);
       if n1==n2&&n1==1
           bSelfConnect=fea_b;
           fea_mean=mean(fea_a,1);
           fea_a=fea_a-repmat(fea_mean, [size(fea_a,1),1]);
           D=EuDist2(fea_a,fea_a,1);
           if bSelfConnect==0
                maxD=max(max(D));
                D=D+2*maxD*eye(size(D,1));
           end
           return ;
       end
    end
    fea_mean=mean([fea_a;fea_b],1);
    fea_a=fea_a-repmat(fea_mean, [size(fea_a,1),1]);
    fea_b=fea_b-repmat(fea_mean, [size(fea_b,1),1]);
    D=EuDist2(fea_a,fea_b,1);
end