% Domain Decomposition Type Preconditioner

testcase=input('Please input testcase ','s');
matfilename=strcat('./',testcase,'.mat');

load(matfilename);
disp('Graph read finished');

LG=Problem.A;
n=size(LG,1);
b=rand(n,1);
[i,j,k]=find(tril(LG,-1));

core=8;k=20;

[p,q1,q2]=mexmetisforDDM(i,j,k,[n],[core]);
LG=LG(p,p);b=b(p,1);

opts.type='ict';opts.droptol=1e-3;opts.shape='upper';
factors=cell(core,1);
p2=1:n;
for i=1:core
    subdomain=q1(i):(q1(i+1)-1);
    B=LG(subdomain,subdomain);
    P=amd(B);
    R=ichol(B(P,P),opts);
    factors(i,:)={R};
    p2(1,subdomain)=P+(q1(i)-1);
end

subdomain=1:(q2(1)-1);
interface=q2(1):n;
C=LG(interface,interface);
P=amd(C);
Rs=chol(C(P,P));
p2(1,interface)=P+(q2(1)-1);
LG=LG(p2,p2);b=b(p2,1);
nnz(Rs)
E=LG(subdomain,interface);

s=length(interface);
disp('Preconditioner Construction finished.');
NNZ=nnz(Rs);
for i=1:core
    R=factors{i,1};
    NNZ=NNZ+nnz(R);
end
disp(['fills:',num2str(NNZ/n)]);

tic;
[x,flag,relres,iter,RESVEC]=pcg(LG,b,1e-3,1000,@(x)DDPre(x,factors,core,q1,E,Rs));
toc;
disp(['SLR: ',num2str(iter)]);

x1=LG\b;
disp(['Max error: ',num2str(norm(x1-x,'inf'))]);


function y=DDPre(x,factors,core,q1,E,Rs)
    x1=x(q1(1):q1(core+1)-1,1);
    y1=zeros(size(x1,1),1);
    for i=1:core
        subdomain=q1(i):q1(i+1)-1;
        R=factors{i,1};
        y1(subdomain,1)=R\(R'\(x1(subdomain,1)));
    end
    y2=x(q1(core+1):end,1)-E'*y1;
    y2=(Rs'\y2);
    y3=Rs\y2;
    y=zeros(size(x,1),1);
    y(q1(core+1):end,1)=y3;
    y1=x1-E*y3;
    for i=1:core
        subdomain=q1(i):q1(i+1)-1;
        R=factors{i,1};
        y(subdomain,1)=R\(R'\(y1(subdomain,1)));
    end
end
