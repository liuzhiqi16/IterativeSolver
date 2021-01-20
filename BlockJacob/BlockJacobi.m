% Block Jacobi Preconditioner

clear;
testcase=input('Please input testcase ','s');
matfilename=strcat('./',testcase,'.mat');
load(matfilename);
disp('Graph read finished');
LG=Problem.A;
n=size(LG,1);
b=rand(n,1);
nparts=8;

opts.type='ict';opts.droptol=1e-3;opts.shape='upper';
[i,j,k]=find(tril(LG,-1));
[p,q1,q2]=mexmetis(i,j,k,[n],[nparts]);
LG=LG(p,p);b=b(p,1);

factors=cell(nparts,3);
NNZ=0;
for i=1:nparts
    subdomain=q1(i):q1(i+1)-1;
    B=LG(subdomain,subdomain);
    P=amd(B);
    R=ichol(B(P,P),opts);
    NNZ=NNZ+nnz(R);
    factors(i,:)={R,subdomain,P};
end
disp(['fills:',num2str(NNZ/n)]);

tic;[x,flag,relres,iter]=pcg(LG,b,1e-3,1000,@(x) BlockJacobiPre(x,factors,nparts));toc;
iter

function y=BlockJacobiPre(x,factors,nparts)
    y=zeros(size(x,1),1);
    for i=1:nparts
        R=factors{i,1};subdomain=factors{i,2};p=factors{i,3};
        xi=x(subdomain,1);xi=xi(p,1);
        y1=R\(R'\xi);
        yi=zeros(size(y1,1),1);
        yi(p,1)=y1;
        y(subdomain,1)=yi;
    end
end