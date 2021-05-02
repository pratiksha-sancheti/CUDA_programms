#include<bits/stdc++.h>

using namespace std;

void vector_sum(vector<float> a,float &cpu_res,int n) {
    for (int i = 0; i < n; i++) {
        cpu_res += a[i];
    }
}

void vector_min(vector<float> a,float &cpu_res,int n){
    cpu_res = INT_MAX;
    for (int i = 0; i < n; i++) {
        cpu_res = min(cpu_res,a[i]);
    }
}

void vector_max(vector<float> a,float &cpu_res,int n){
    cpu_res = INT_MIN;
    for (int i = 0; i < n; i++) {
        cpu_res = max(cpu_res,a[i]);
    }
}

void vector_sd(vector<float> a,float sum,double &cpu_res_sd,int n){
    double mean = (double)sum/(double)n;

    double s = 0;
    for(int i=0;i<n;i++){
        s += ((a[i]-mean)*(a[i]-mean));
    }

    cpu_res_sd = (double)s/(double)n;
}

__global__ void cuda_vector_sum(float* a,int n) {
    const int tid=threadIdx.x;
    int no_of_threads=blockDim.x;

    for(int step=1;step < n; step *= 2,no_of_threads /= 2){
      if (tid <= no_of_threads){
        int ind=2*step*tid;
      
        if((ind+step) >= n){
          a[ind] = a[ind] + 0;
        }else{
          a[ind] = a[ind] + a[ind+step];
        }
      }
    }
}

__global__ void cuda_vector_min(float* a,int n) {
    const int tid=threadIdx.x;
    int no_of_threads=blockDim.x;

    for(int step=1;step < n; step *= 2,no_of_threads /= 2){
      if (tid <= no_of_threads){
        int ind=2*step*tid;
      
        if((ind+step) >= n){
          a[ind] = min(a[ind],FLT_MAX);
        }else{
          a[ind] = min(a[ind],a[ind+step]);
        }
      }
    }
}

__global__ void cuda_vector_max(float* a,int n) {
    const int tid=threadIdx.x;
    int no_of_threads=blockDim.x;

    for(int step=1;step < n; step *= 2,no_of_threads /= 2){
      if (tid <= no_of_threads){
        int ind=2*step*tid;
      
        if((ind+step) >= n){
          a[ind] = max(a[ind],FLT_MIN);
        }else{
          a[ind] = max(a[ind],a[ind+step]);
        }
      }
    }
}

__global__ void cuda_update_arr(float *a,double mean){
    const int tid=threadIdx.x;
    a[tid] = (a[tid]-mean)*(a[tid]-mean);
}

int main() {
    int N = 2048;

    vector<float> a(N);
    srand(time(0));
    generate(begin(a), end(a), []() { return (float(rand())/float((RAND_MAX)) * 100.0); });
  
    for(auto item:a)
      cout<<item<<" ";
    cout<<'\n';

    float cpu_res=0,gpu_res=0;
    double cpu_res_sd = 0,gpu_res_sd = 0;

    cout<<"CPU: "<<'\n';
  //-------------------------------------------------------------------
    
    // Sum calculation
    vector_sum(a,cpu_res,N);
    cout << "Vector Sum using CPU :"<<cpu_res<<" \n";

    // Average calculation
    cout << "Vector Average using CPU :"<<(double)cpu_res/(double)N<<" \n";

    vector_sd(a,cpu_res,cpu_res_sd,N);
    cout << "Vector Standard Deviation using CPU :"<<fixed<<setprecision(2)<<sqrt(cpu_res_sd)<<" \n";

    vector_min(a,cpu_res,N);
    cout << "Vector Min using CPU :"<<cpu_res<<" \n";

    vector_max(a,cpu_res,N);
    cout << "Vector Max using CPU :"<<cpu_res<<" \n";
    

  cout<<"GPU: "<<'\n';
  //-------------------------------------------------------------------
    // Allocate memory on the device
    size_t bytes = sizeof(float) * N;
    float* d_a;
    cudaMalloc(&d_a, bytes);
  //-------------------------------------------------------------------

    // Copy data from the host to the device (CPU to GPU)
   
    

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cuda_vector_sum <<<1,N/2>>> (d_a,N);
    cudaMemcpy(&gpu_res, d_a, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Vector Sum using GPU :"<<gpu_res<<" \n";

//-------------------------------------------------------------------

    cout << "Vector Average using GPU :"<<(double)gpu_res/(double)N<<" \n";

//-------------------------------------------------------------------

    double mean = (double)gpu_res/(double)N;
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cuda_update_arr<<<1,N>>>(d_a,mean);
    cuda_vector_sum<<<1,N/2>>>(d_a,N);
    cudaMemcpy(&gpu_res, d_a, sizeof(float), cudaMemcpyDeviceToHost);
    gpu_res = (double)gpu_res/(double)N;
    cout << "Vector Standard Deviation using GPU :"<<fixed<<setprecision(2)<<sqrt(gpu_res)<<" \n";
    

//-------------------------------------------------------------------

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    gpu_res = INT_MAX;
    cuda_vector_min <<<1,N/2>>> (d_a,N);
    cudaMemcpy(&gpu_res, d_a, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Vector Min using GPU :"<<gpu_res<<" \n";

//-------------------------------------------------------------------

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    gpu_res = INT_MIN;
    cuda_vector_max <<<1,N/2>>> (d_a,N);
    cudaMemcpy(&gpu_res, d_a, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Vector Max using GPU :"<<gpu_res<<" \n";

//-------------------------------------------------------------------

    // Free memory on device
    cudaFree(d_a);
}