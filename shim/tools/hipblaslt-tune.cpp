#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  int S = argc > 1 ? atoi(argv[1]) : 512;
  hipblasLtHandle_t lt; hipblasLtCreate(&lt);
  hipStream_t stream; hipStreamCreate(&stream);

  void *A,*B,*C,*WS;
  size_t bytes = (size_t)S*S*4;
  hipMalloc(&A,bytes); hipMalloc(&B,bytes); hipMalloc(&C,bytes);
  size_t wsz = 128ull*1024*1024; hipMalloc(&WS,wsz);
  hipMemset(A,0,bytes); hipMemset(B,0,bytes);

  hipblaslt_ext::Gemm gemm(lt, HIPBLAS_OP_N, HIPBLAS_OP_N,
                           HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F,
                           HIPBLAS_COMPUTE_32F);
  hipblaslt_ext::GemmEpilogue epi;
  hipblaslt_ext::GemmInputs in;
  float alpha=1.0f, beta=0.0f;
  in.setA(A); in.setB(B); in.setC(C); in.setD(C);
  in.setAlpha(&alpha); in.setBeta(&beta);
  if (gemm.setProblem(S,S,S,1,epi,in) != HIPBLAS_STATUS_SUCCESS) {
    printf("setProblem failed\n"); return 1;
  }

  hipblaslt_ext::GemmPreference pref;
  pref.setMaxWorkspaceBytes(wsz);
  std::vector<hipblasLtMatmulHeuristicResult_t> res;
  gemm.algoGetHeuristic(32, pref, res);
  printf("%d^3 fp32: %zu algos\n", S, res.size());

  hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
  const int REPS=20;
  double flops = 2.0*S*S*S;
  double best=-1; int bA=-1,bK=-1,bW=-1;

  for (size_t i=0;i<res.size();++i) {
    for (int sk : {0,1,2,4,8,16,32}) {
      for (int wg : {0,1,2,4,8,16}) {
        hipblaslt_ext::GemmTuning tune;
        tune.setSplitK((uint16_t)sk); tune.setWgm((int16_t)wg);
        size_t need=0;
        if (gemm.isAlgoSupported(res[i].algo, tune, need)!=HIPBLAS_STATUS_SUCCESS) continue;
        if (need>wsz) continue;
        if (gemm.initialize(res[i].algo, tune, WS, true, stream)!=HIPBLAS_STATUS_SUCCESS) continue;
        if (gemm.run(stream)!=HIPBLAS_STATUS_SUCCESS) continue;
        hipStreamSynchronize(stream);
        hipEventRecord(e0,stream);
        for(int r=0;r<REPS;++r) gemm.run(stream);
        hipEventRecord(e1,stream); hipEventSynchronize(e1);
        float ms=0; hipEventElapsedTime(&ms,e0,e1);
        double g = flops/((ms/REPS)*1e6);
        if (g>best){best=g;bA=(int)i;bK=sk;bW=wg;}
      }
    }
  }
  {
    hipblaslt_ext::GemmTuning t; t.setSplitK((uint16_t)bK); t.setWgm((int16_t)bW);
    size_t need=0; gemm.isAlgoSupported(res[bA].algo, t, need);
    printf("BEST %.0f GFLOP/s  algo=%d splitK=%d wgm=%d  workspace_needed=%zu bytes (%.2f MB)\n",
           best,bA,bK,bW,need,need/1048576.0);
  }
  return 0;
}
