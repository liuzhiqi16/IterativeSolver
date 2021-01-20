# include "metis.h"
# include <mex.h>
# include <iostream>
# include <fstream>
# include <vector>
# include <algorithm>
# include <cmath>
# include <stdlib.h>
# include <cstring>
# include <queue>
# include <ctime>
# include <string>
# include <list>
# include <stack>

using namespace std;

class Graph{
    public:
		int* xadj;
        int* adjncy;
};

int *ai,*aj;
double* av;
int M,SIZE;
Graph gra;

void constructgraph(){
    gra=Graph();
	gra.adjncy=(int*) malloc(sizeof(int)*(2*M));
	gra.xadj=(int*) malloc(sizeof(int)*(SIZE+1));
	int* ptrs=(int*) malloc(sizeof(int)*(SIZE));
    int* deg=(int*) malloc(sizeof(int)*(SIZE));memset(deg,0,sizeof(int)*(SIZE));
    int i,j;
	for(int l=0;l<M;l++){
		i=ai[l];j=aj[l];
		deg[i]++;deg[j]++;
	}
    gra.xadj[0]=0;ptrs[0]=0;
	int sum=0;
	for(int l=1;l<SIZE;l++){
		sum+=deg[l-1];
		gra.xadj[l]=sum;
		ptrs[l]=sum;
	}
	sum+=deg[SIZE-1];
	gra.xadj[SIZE]=sum;
	for(int l=0;l<M;l++){
		i=ai[l];j=aj[l];
		gra.adjncy[ptrs[i]]=j;
		gra.adjncy[ptrs[j]]=i;
		ptrs[i]++;ptrs[j]++;
	}
	free(ptrs);free(deg);free(ai);free(aj);
}

bool isinterface(int i,int* part){
	for(int k=gra.xadj[i];k<gra.xadj[i+1];k++){
		if(part[gra.adjncy[k]]!=part[i]){
			return true;
		}
	}
	return false;
}

//[p,q1,q2]=mexmetis(i,j,k,n,nparts)
void mexFunction(
    int	nargout,
    mxArray *pargout [],
    int	nargin,
    const mxArray *pargin []
){
    double *pi,*pj,*pk,*pn,*pnparts;
    pi=mxGetPr(pargin[0]);
    pj=mxGetPr(pargin[1]);
    pk=mxGetPr(pargin[2]);
	pn=mxGetPr(pargin[3]);
	pnparts=mxGetPr(pargin[4]);
	int nparts=(int)(pnparts[0]);
    M=mxGetNumberOfElements(pargin[0]);
    SIZE=(int)(pn[0]);
	pargout[0]=mxCreateDoubleMatrix(SIZE,1,mxREAL);
	pargout[1]=mxCreateDoubleMatrix(nparts+1,1,mxREAL);
	pargout[2]=mxCreateDoubleMatrix(nparts+1,1,mxREAL);
	double* ppi=mxGetPr(pargout[0]);
	double* pptrs=mxGetPr(pargout[1]);
	double* pptri=mxGetPr(pargout[2]);
    ai=(int*) malloc(sizeof(int)*(M+10));
    aj=(int*) malloc(sizeof(int)*(M+10));
    for(int i=0;i<M;i++){
        ai[i]=(int)pi[i]-1;aj[i]=(int)pj[i]-1;
    }
    constructgraph();
    //cout<<"read finished."<<endl;
	//cout<<M<<","<<SIZE<<endl;
	
    int nvtx=SIZE;int ncon=1;double ubvec[]={1.5};
    int totv=0;
    int* part=(int*) malloc(sizeof(int)*(SIZE));

    int options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING]=0;

    clock_t begin=clock();
    int res=METIS_PartGraphRecursive(&nvtx,&ncon,gra.xadj,gra.adjncy,NULL,NULL,NULL,&nparts,NULL,NULL,options,&totv,part);
    clock_t end=clock();
    if(res==METIS_OK){
  //      cout<<"Succeed."<<endl;
  //      cout<<"Interface nodes: "<<totv<<endl;
  //      cout<<"Metis time: "<<(double)(end-begin)/CLOCKS_PER_SEC<<endl;    
    }
    int* nums=(int*) malloc(sizeof(int)*(nparts+1));
    memset(nums,0,sizeof(int)*(nparts+1));
	int* numi=(int*) malloc(sizeof(int)*(nparts+1));
    memset(numi,0,sizeof(int)*(nparts+1));
	int nvtxsub=0;
    for(int i=0;i<SIZE;i++){
		nums[part[i]+1]++;
    }
	nums[0]=0;
	pptrs[0]=1+nums[0];
	for(int i=1;i<=nparts;i++){
		nums[i]=nums[i]+nums[i-1];
		numi[i-1]=nums[i];
		pptrs[i]=nums[i]+1;
	}
	for(int i=0;i<SIZE;i++){
		if(!isinterface(i,part)){
			ppi[nums[part[i]]]=(i+1);
			nums[part[i]]+=1;
		}
		else{
			numi[part[i]]-=1;
			ppi[numi[part[i]]]=(i+1);
		}
	}
	for(int i=0;i<nparts;i++){
		pptri[i]=numi[i]+1;
	}
}
