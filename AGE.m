function [S,gamma] = AGE(options)
%% Adaptive Graph Embedding (AGE)
%%% calculate adaptive gamma and beta by Theorem 1.
%% input:
%%% dist: the geometric metric of the sample X;
%%% semanticGraph: the semantic metric of the sample X;
%%% tau: an arbitrary value used to emphasize the semantic information;
%%% rr:  an adaptive parameter (gamma in the paper);
%%% k:         the number of neighbors
%% output:
%%% S:         the similar matrix n2*n1
    %% ===== init parameters =====
    if ~isfield(options,'G')
        options.G = -1;
    end
    if ~isfield(options,'slack')
        options.slack = 0;
    end
    dist=options.dist;
    G=options.semanticGraph;
    tau=options.tau ; % a
    k=options.k;
    [n2,n1]=size(dist);
    %% init sorted semantic graph
    
    % relax the semantic graph to achieve discriminative results if necessary.
    if options.slack==1
        G(G==0)=-1;
    end
    [d, idx] = sort(dist,2,'ascend');
    sortG=zeros(n2,n1);
    for i=1:n2
        sortG(i,:)=G(i,idx(i,:));
    end
    
    if ~isfield(options,'rr')
        %% calculate beta_i by Eq. (13)
        [d_res,g_res]=calculateDifference(d,sortG,k,k); % calculate k-th term
        [d_res2,g_res2]=calculateDifference(d,sortG,k,k+1); % calculate (k+1)-th term
        tmp1=d_res-d_res2;
        tmp2=g_res-g_res2;
        beta_i=-tmp1./tmp2;
        beta_i(tmp2==0)=tau;
        %% calculate gamma_i by Eq.(14)
        gamma_i=calculateGamma(d,sortG,beta_i,k);
        gamma=mean(gamma_i);
        %% calculate S by Eq. (32)
        S = zeros(n2,n1);
        S_sup=getSimilarity(d,sortG,beta_i);
        for i = 1:n2
            idxa0 = idx(i,1:k);
            tmp=(S_sup(i,1:k))./(gamma_i(i)+beta_i(i));
            [v,~]=EProjSimplex_new( tmp );
            S(i,idxa0)=v;
        end
    else
        rr=options.rr; % use rr to replace gamma
        %% calculate beta_i by Eq. (13)
        [d_res,g_res]=calculateDifference(d,sortG,k,k);
        [d_res2,g_res2]=calculateDifference(d,sortG,k,k+1);
        tmp1=d_res-d_res2; % 
        tmp2=g_res-g_res2; % -2k, 0, 2k
        beta_i=-tmp1./tmp2;
        beta_i(tmp2==0)=tau;
        %% calculate gamma_i by Eq.(14)
        gamma_i=calculateGamma(d,sortG,beta_i,k);
        gamma=mean(gamma_i);
        S = zeros(n2,n1);
        %% calculate S by Eq. (32)
        S_sup=getSimilarity(d,sortG,beta_i);
        for i = 1:n2
            idxa0 = idx(i,1:k);
            tmp=S_sup(i,1:k)./(rr+beta_i(i));
            [v,~]=EProjSimplex_new(tmp);
            S(i,idxa0)=v;
        end
    end
    
end
function S=getSimilarity(D,G,beta)
    %% S=-1/2 d_{i,j}+\beta_i * g_{i,j}
    if length(beta)>1
        S=-0.5*D+repmat(beta,[1 size(G,2)]).*G;
    else
        S=-0.5*D+beta*G;
    end
end
function gamma_i=calculateGamma(D,Y,beta_i,k)
    %% calculate gamma_i by Eq. (45)
    [d_k_res,g_k_res]=calculateDifference(D,Y,k,k); % k-th term
    d_k_res=-d_k_res;
    g_k_res=-g_k_res;
    idx=g_k_res>=0;
    f_k_pos=d_k_res+beta_i.*g_k_res - beta_i;
    f_k_neg=d_k_res+beta_i.*g_k_res - beta_i;
    gamma_i=f_k_pos.*idx+f_k_neg.*(1-idx);
end

function [d_res,g_res]=calculateDifference(D,G,k,index)
%%% calculate the difference between top-k samples and the rest of the samples.
%%%      under geometric and semantic metrics
%% input:
%%% D:      the geometric metric;
%%% G:      the semantic metric;
%%% k:      the neighborhood number;
%%% index:  to indicate calculate k-th term or (k+1)-th term;
%% output:
%%% d_res:  the difference of the geometry
%%% g_res:  the difference of the semantics

    %% function A in the paper, Eq. (38) 
    d_res=-0.5*(k*D(:,index)-sum(D(:,1:k),2)); 
    %% function B in the paper, Eq. (39)
    g_res=k*G(:,index)-sum(G(:,1:k),2);
end

function [x,ft] = EProjSimplex_new(v, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%
if nargin < 2
    k = 1;
end
ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);
else
    x = v0;
end
end