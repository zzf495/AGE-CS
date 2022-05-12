function [Yt,f,Z,opt] = SRM(Xs,Xt,Ys,options)
%% Structure Risk Minimization (SRM)
%%% Input:
%%% Xs (m*ns): training samples
%%% Xt (m*nt): test samples
%%% Ys (ns*1): the labels of training samples
%%% options.Kernel: the kernel function, linear/1, rbf/2, sam/3
%%% options.gamma: the kernel parameter
%%% options.mu: the regularization parameter
%%% options.accelerate (optional*): default 0, if it >0, kernel will be
%%%                                         constructed by anchors A, approix
%%%                                         by \phi(X,A) phi(A,A)^{-1}\phi(A,X)
%%% options.anchors (optional*): the anchors
%%% Output:
%%% f (m*C): learned projection matrix
%%% Yt (nt*1): the predicted labels
%%% Z (C*n): 
    X=[Xs,Xt];
    ns=size(Xs,2);
    nt=size(Xt,2);
    C=length(unique(Ys));
    Yt=[];
    options=defaultOptions(options,'Kernel',2,...
        'gamma',0.1,...
        'mu',0.1,...
        'accelerate',0);
    Kernel=options.Kernel;
    gamma=options.gamma;
    mu=options.mu;
    
    if Kernel~=0
        if options.accelerate>0
            anchors=options.anchors;
            Kernel_anchors = kernelProject(Kernel,anchors,anchors,gamma);
            Kernel_XA = kernelProject(Kernel,X,anchors,gamma);
            X=Kernel_XA*inv(Kernel_anchors+eye(size(anchors,2)))*Kernel_XA';
        else
            X = kernelProject(Kernel,X,[],gamma);
        end
        E=diag([ones(ns,1);zeros(nt,1)]);
        Y=[hotmatrix(Ys,C);zeros(nt,C)];
        m=size(X,1);
        f=((E*X)+mu*eye(m))\(E*Y);
    else
        m=size(X,1);
        E=diag([ones(ns,1);zeros(nt,1)]);
        Y=[hotmatrix(Ys,C);zeros(nt,C)];
        f=((X*E*X')+mu*eye(m))\(X*E*Y);
    end
    Z=f'*X;
    [~,Yt]=max(Z(:,ns+1:end),[],1);
    Yt=Yt';
    opt=options;
end
