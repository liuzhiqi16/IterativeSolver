% Multi level low rank correction preconditioner
% Serial version
% MATLAB implementation of the paper "Divide and conquer low-rank preconditioners for symmetric matrices".


testcase=input('Please input testcase ','s');
matfilename=strcat('./',testcase,'.mat');
load(matfilename);
disp('Graph read finished');
LG=Problem.A;
n=size(LG,1);
b=rand(n,1);

nlevel=3;rank=4;
B=LG;
leaf=cell(2^nlevel,2); % Store R and P;
nonleaf=cell(2^nlevel-1,4); % Store P, E, U and H
offsets=zeros(2^(nlevel+1)-1,1);
lengths=zeros(2^(nlevel+1)-1,1);
offsets(1)=0;lengths(1)=n;

% 从上到下，计算Bi和Ei
for index=1:(2^nlevel-1)
    rangei=1:lengths(index);
    rangei=rangei+offsets(index);
    Bi=B(rangei,rangei);
    [i,j,k]=find(tril(Bi,-1));
    [p,q1,q2]=mexmetis(i,j,k,[lengths(index)],[2]);
    Bi=Bi(p,p);
    nonleaf(index,1)={p};
    W=-Bi(q2(1):q1(2)-1,q2(2):q1(3)-1);
    [m,mm]=size(W);
    X1=zeros(m,1);
    for jj=1:m
        X1(jj)=sqrt(norm(W(jj,:)));
        W(jj,:)=W(jj,:)/X1(jj);
    end
    X1=spdiags(X1,[0],m,m);
    X2=W;
    E=[zeros(q2(1)-q1(1),m);X1;zeros(q2(2)-q1(2),m);X2'];
    nonleaf(index,2)={E};
    B(rangei,rangei)=Bi+E*E';
    l=lchild(index);r=rchild(index);
    lengths(l)=q1(2)-q1(1);lengths(r)=q1(3)-q1(2);
    offsets(l)=offsets(index);offsets(r)=offsets(index)+lengths(l);
end
   
% 对于叶子节点，分解Bi
for index=(2^nlevel):(2^(nlevel+1)-1)
    rangei=1:lengths(index);
    rangei=rangei+offsets(index);
    Bi=B(rangei,rangei);
    p=amd(Bi);
    R=chol(Bi(p,p));
    leaf(index-2^nlevel+1,1)={R};
    leaf(index-2^nlevel+1,2)={p};
end

% 从下向上，计算U和H
for index=(2^nlevel-1):-1:1
    E=nonleaf{index,2};
    [U,S,V]=svds(@(b,tflag)Afun(b,tflag,index,leaf,nonleaf,lengths,nlevel),size(E),rank);
    U=U*S;
    V=V(:,1:rank);
    H=inv(eye(rank)-U'*E*V);
    nonleaf(index,3)={U};
    nonleaf(index,4)={H};
end

disp('Preconditioner construction finished.');
% PCG
tic;[x,flag,relres,iter,RESVEC]=pcg(LG,b,1e-3,1000,@(x)MLRSolve(x,1,leaf,nonleaf,lengths,nlevel));toc;
disp(['MLR: ',num2str(iter)]);

function l=lchild(n)
    l=2*n;
end
function r=rchild(n)
    r=2*n+1;
end
function p=parent(n)
    p=floor(n/2);
end

% 预条件子求解，多级递归求解
function y=MLRSolve(x,index,leaf,nonleaf,lengths,nlevel)
    y=zeros(size(x,1),1);    
    if(index>=(2^nlevel))
    % leaf node
        p=leaf{index-2^nlevel+1,2};
        R=leaf{index-2^nlevel+1,1};
        y(p,1)=R\(R'\(x(p,1)));
    else
    % nonleaf node
        p=nonleaf{index,1};
        U=nonleaf{index,3};
        H=nonleaf{index,4};
        lengthl=lengths(lchild(index));lengthr=lengths(rchild(index));
        x1=x(p,1);
        y1=MLRSolve(x1(1:lengthl,1),lchild(index),leaf,nonleaf,lengths,nlevel);
        y2=MLRSolve(x1(lengthl+1:end,1),rchild(index),leaf,nonleaf,lengths,nlevel);
        y3=[y1;y2]+U*(H*(U'*x1));
        y(p,1)=y3;
    end
end

function y=aprxinvB(x,index,leaf,nonleaf,lengths,nlevel)
    lengthl=lengths(lchild(index));
    y1=MLRSolve(x(1:lengthl,1),lchild(index),leaf,nonleaf,lengths,nlevel);
    y2=MLRSolve(x(lengthl+1:end,1),rchild(index),leaf,nonleaf,lengths,nlevel);
    y=[y1;y2];
end

function y=Afun(b,tflag,index,leaf,nonleaf,lengths,nlevel)
    E=nonleaf{index,2};
    if strcmp(tflag,'notransp')
        x=E*b;
        y=aprxinvB(x,index,leaf,nonleaf,lengths,nlevel);
    else
        x=aprxinvB(b,index,leaf,nonleaf,lengths,nlevel);
        y=E'*x;
    end
end