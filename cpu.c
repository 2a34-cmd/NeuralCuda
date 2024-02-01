#include <stdio.h>
#include <stdlib.h>

#define N 10000

int main(){
    int a[N],b[N],c[N];

    for(int i=0; i <N; i++){
        a[i]= -i;
        b[i]= i*i;
    }

    for(int j=0; j <N; j++){

        c[j]= a[j]+b[j];
    }

    for(int k =0 ; k<N; k+=100){
        printf("%d + %d = %d \n",a[k],b[k],c[k]);
    }
}