// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "spu/mpc/semi2k/arithmetic.h"

#include "spu/core/profile.h"
#include "spu/core/vectorize.h"
#include "spu/mpc/common/abprotocol.h"  // zero_a
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/semi2k/object.h"
#include "spu/mpc/semi2k/type.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"
#include <time.h>
#include <float.h>
#include <iostream>
#include <fstream>
#include <string>

// void writeFile(const spu::ArrayRef& x) {
//   // std::ofstream write;
//   // std::ifstream read;
//   std::fstream write;
//   write.open("/home/admin/dev/spu/mpc/semi2k/1000-1000-ss.txt",std::ios::out|std::ios::app);
//   if(!write) {
//     std::cout<<"open file failed"<<std::endl;
//     return;
//   }
//   for(int64_t i = 0; i < x.numel(); i++) {
//     // std::cout<<x.at<int32_t>(i)<<std::endl;
//     write<<x.at<int32_t>(i)<<std::endl;
//   }
//   write.close();
//   // read.close();
// }


static double onlineTime = 0.0;
// static double offlineTime = 0.0;
static double onlineMmul = 0.0;
// static double offlineMmul = 0.0;
clock_t start_online_mmul, end_online_mmul;
clock_t start_online, end_online;

static double total_bytes = 0.0;

// static void setRunningTime(double newOffline, double newOnline) {
//   offlineTime += newOffline;
//   onlineTime += newOnline;
// }



namespace spu::mpc::semi2k {

ArrayRef ZeroA::proc(KernelEvalContext* ctx, FieldType field,
                     size_t size) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, size);

  auto* prg_state = ctx->caller()->getState<PrgState>();

  auto [r0, r1] = prg_state->genPrssPair(field, size);
  return ring_sub(r0, r1).as(makeType<AShrTy>(field));
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  auto x = zero_a(ctx->caller(), field, in.numel());

  if (comm->getRank() == 0) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::ADD, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();

  // First, let's show negate could be locally processed.
  //   let X = sum(Xi)     % M
  //   let Yi = neg(Xi) = M-Xi
  //
  // we get
  //   Y = sum(Yi)         % M
  //     = n*M - sum(Xi)   % M
  //     = -sum(Xi)        % M
  //     = -X              % M
  //
  // 'not' could be processed accordingly.
  //   not(X)
  //     = M-1-X           # by definition, not is the complement of 2^k
  //     = neg(X) + M-1
  //
  auto res = ring_neg(in);
  if (comm->getRank() == 0) {
    const auto field = in.eltype().as<Ring2k>()->field();
    ring_add_(res, ring_not(ring_zeros(field, in.numel())));
  }

  return res.as(in.eltype());
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  YASL_ENFORCE(lhs.numel() == rhs.numel());
  auto* comm = ctx->caller()->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_add(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  YASL_ENFORCE(lhs.numel() == rhs.numel());
  YASL_ENFORCE(lhs.eltype() == rhs.eltype());

  return ring_add(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  return ring_mul(lhs, rhs).as(lhs.eltype());
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();
  auto [a, b, c] = beaver->Mul(field, lhs.numel());

  // Open x-a & y-b
  auto res =
      vectorize({ring_sub(lhs, a), ring_sub(rhs, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kBindName);
      });

  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_add(ring_mul(x_a, b), ring_mul(y_b, a)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, y);
  return ring_mmul(x, y, M, N, K).as(x.eltype());
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, y);

  // constexpr char kLinkAddrAB[] = "127.0.0.1:9532,127.0.0.1:9533";
  // std::cout<<" M = "<<M<<", N = "<<N<<", K = "<<K<<std::endl;

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  // start_offline_mmul = clock();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

  // generate beaver multiple triple.
  auto [a, b, c] = beaver->Dot(field, M, N, K);
  // end_offline_mmul = clock();
  start_online_mmul = clock();


  // Open x-a & y-b
  // auto res =
  //     vectorize({ring_sub(x, a), ring_sub(y, b)}, [&](const ArrayRef& s) {
  //       return comm->allReduce(ReduceOp::ADD, s, kBindName);
  //     });
  // auto x_a = std::move(res[0]);
  // auto y_b = std::move(res[1]);
  // writeFile(ring_sub(x,a));
  // writeFile(ring_sub(y,b));
  
  auto x_a = comm->allReduce(ReduceOp::ADD, ring_sub(x, a), kBindName);
  auto y_b = comm->allReduce(ReduceOp::ADD, ring_sub(y, b), kBindName);

  // std::cout<<"x-a.numel()"<<ring_sub(x,a).numel()<<std::endl;
  // std::cout<<"y-b.numel()"<<ring_sub(y,b).numel()<<std::endl;
  

  end_online_mmul = clock();
  // offlineMmul += double(end_offline_mmul - start_offline_mmul)/CLOCKS_PER_SEC; 
  onlineMmul += double(end_online_mmul - start_online_mmul)/CLOCKS_PER_SEC;
  // setRunningTime(offline, online);
  // std::cout << "mmul offline time = "<<offlineMmul<<"s"<<std::endl; 
  std::cout << "mmul online time = "<<onlineMmul<<"s"<<std::endl; 

  // auto stats0 = ctx->GetStats();
  // auto stats1 = ctx[1]->GetStats();
  // if(comm->getRank() == 1) {
  //   auto stats0 = comm->getStats();

  //   total_bytes += stats0.comm;
  //   SPDLOG_INFO("total_comm_bytes:{}",
  //               // stats1->sent_bytes, stats1->recv_bytes,
  //               total_bytes/1024/1024);
  // }
  

  // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
  auto z = ring_add(
      ring_add(ring_mmul(x_a, b, M, N, K), ring_mmul(a, y_b, M, N, K)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mmul(x_a, y_b, M, N, K));
  }

  return z.as(x.eltype());
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  return ring_lshift(in, bits).as(in.eltype());
}

ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, bits);
  auto* comm = ctx->caller()->getState<Communicator>();

  // TODO: add trunction method to options.
  if (comm->getWorldSize() == 2u) {
    // SecurlML, local trunction.
    // Ref: Theorem 1. https://eprint.iacr.org/2017/396.pdf
    return ring_arshift(x, bits).as(x.eltype());
  } else {
    // ABY3, truncation pair method.
    // Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
    auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

    const auto field = x.eltype().as<Ring2k>()->field();
    const auto& [r, rb] = beaver->Trunc(field, x.numel(), bits);

    // open x - r
    auto x_r = comm->allReduce(ReduceOp::ADD, ring_sub(x, r), kBindName);
    auto res = rb;
    if (comm->getRank() == 0) {
      ring_add_(res, ring_arshift(x_r, bits));
    }

    // res = [x-r] + [r], x which [*] is truncation operation.
    return res.as(x.eltype());
  }
}

static void TransposeInplace(ArrayRef mat, size_t nrows, size_t ncols) {
  YASL_ENFORCE_EQ((size_t)mat.numel(), nrows * ncols);
  const auto field = mat.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto xmat = xt_mutable_adapt<ring2k_t>(mat);
    xmat.reshape({nrows, ncols});
    auto xmatT = xt::eval(xt::transpose(xmat));
    std::copy_n(xmatT.begin(), xmatT.size(), xmat.data());
  });
}



////////////////////////////////////////////////////////////////////
// lj : logreg family
////////////////////////////////////////////////////////////////////
ArrayRef LogRegAP::proc(KernelEvalContext* ctx, const ArrayRef& x, const ArrayRef& w,
                        const ArrayRef& y, size_t K, size_t N, size_t M) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, w, y);
  
  auto x_ = x.clone();
  TransposeInplace(x_, M, N);

  auto y_pred = ring_mmul(w, x_, K, M, N);

  auto err = ring_sub(y_pred, y);

  auto grad = ring_mmul(err, x, K, N, M);

  return grad.as(x.eltype());
}

ArrayRef LogRegAA::proc(KernelEvalContext* ctx, const ArrayRef& x, const ArrayRef& w,
                        const ArrayRef& y, size_t K, size_t N, size_t M) const {

  SPU_PROFILE_TRACE_KERNEL(ctx, x, w, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  // start_offline = clock();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();
  auto [r1, r2, r3, c1, c3, c4, c5] = beaver->lr(field, M, K, N);

  // writeFile(x);
  // writeFile(w);
  // // writeFile(y);

  // end_offline = clock();
  auto x_ = ring_sub(x,r1);
  auto y_ = ring_sub(y,r3);
  auto w_ = ring_sub(w,r2);

  start_online = clock();
  auto x_r1 = comm->allReduce(ReduceOp::ADD, x_, kBindName);
  auto w_r2 = comm->allReduce(ReduceOp::ADD, w_, kBindName);
  auto y_r3 = comm->allReduce(ReduceOp::ADD, y_, kBindName);
  end_online = clock();
  // offlineTime += double(end_offline - start_offline)/CLOCKS_PER_SEC; 
  onlineTime += double(end_online - start_online)/CLOCKS_PER_SEC;
  // setRunningTime(offline, online);
  // std::cout << "offline time = "<<offlineTime<<"s"<<std::endl; 
  std::cout << "online time = "<<onlineTime<<"s"<<std::endl; 

  auto stats0 = comm->getStats();

  total_bytes += stats0.comm;
  SPDLOG_INFO("total_comm_bytes:{}",
                // stats1->sent_bytes, stats1->recv_bytes,
                total_bytes/1024/1024);

  // auto x_r1 = comm->allReduce(ReduceOp::ADD, x, kBindName);
  // auto y_r3 = comm->allReduce(ReduceOp::ADD, y, kBindName);
  // auto w_r2 = comm->allReduce(ReduceOp::ADD, w, kBindName);

  //1024-the transpose of x_r1
  auto x_r1T = x_r1.clone();
  TransposeInplace(x_r1T, K, M);

  // //1024-the transpose of r1
  auto r1T = r1.clone();
  TransposeInplace(r1T, K, M);

  // auto iden2 = ring_add(ring_ones(field, K * M), ring_ones(field, K * M));

  //1024-lj-transpose
  auto y_r3T = y_r3.clone();
  TransposeInplace(y_r3T, K, N);

  auto r3T = r3.clone();
  TransposeInplace(r3T, K, N);

  // auto two = ring_add(ring_ones(field, K * M), ring_ones(field, K * M));
  // auto four = ring_add(two, two);
  // auto eight = ring_add(four, four);

  // size_t bits = 18;

  // auto a1 = ring_arshift(ring_mmul(iden2, r1, K, N, M), bits);
  // auto a2 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(w_r2, r1T, K, M, N), bits), x_r1, K, N, M), bits);
  // auto a3 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(r2, x_r1T, K, M, N), bits), x_r1, K, N, M), bits);
  // auto a4 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(w_r2, x_r1T, K, M, N), bits), r1, K, N, M), bits);
  // auto a5 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(w_r2, r1T, K, M, N), bits), r1, K, N, M), bits);
  // auto a6 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(r2, x_r1T, K, M, N), bits), r1, K, N, M), bits);
  // auto a7 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(r2, r1T, K, M, N), bits), r1, K, N, M), bits);
  // auto a14 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(r2, r1T, K, M, N), bits), x_r1, K, N, M), bits);
  // auto a8 = ring_arshift(ring_mul(four, ring_arshift(ring_mmul(y_r3T, r1, K, N, M), bits)), bits);
  // auto a9 = ring_arshift(ring_mul(four, ring_arshift(ring_mmul(r3T, x_r1, K, N, M), bits)), bits);
  // auto a10 = ring_arshift(ring_mul(four, ring_arshift(ring_mmul(r3T, r1, K, N, M), bits)), bits);
  // auto grad = ring_sub(ring_sub(ring_sub(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(
  //   a1, a2), a3), a4), a5), a6), a7), a14), a8), a9), a10);

  size_t bits = 18;
  auto iden2 = ring_add(ring_ones(field, N * K), ring_ones(field, N * K));
  auto iden4 = ring_add(iden2, iden2);
  auto iden8 = ring_add(iden4, iden4);

  auto two = ring_add(ring_ones(field, N * M), ring_ones(field, N * M));
  auto four = ring_add(two, two);
  auto eight = ring_add(four, four);

  //ADD
  auto a1 = ring_mmul(iden2, r1, N, M, K);
  auto a2 = ring_arshift(ring_mmul(ring_mmul(w_r2, r1T, N, K, M), x_r1, N, M, K), bits);
  auto a3 = ring_arshift(ring_mmul(ring_mmul(r2, x_r1T, N, K, M), x_r1, N, M, K), bits);
  auto a4 = ring_arshift(ring_mmul(c1, x_r1, N, M, K), bits);
  auto a5 = ring_arshift(ring_mmul(ring_mmul(w_r2, x_r1T, N, K, M), r1, N, M, K), bits);
  auto a6 = ring_arshift(ring_mmul(w_r2, c5, N, M, M), bits);
  auto a7 = ring_arshift(ring_mmul(ring_mmul(r2, x_r1T, N, K, M), r1, N, M, K), bits);
  auto a8 = c3;
  //SUB
  auto a9 = ring_mul(four, ring_mmul(y_r3T, r1, N, M, K)); //no trunc
  auto a10 = ring_mul(four, ring_mmul(r3T, x_r1, N, M, K)); //no trunc
  auto a11 = c4; //no trunc

  auto add_tmp = ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(a1, a2), a3), a4), a5), a6), a7), a8);
  auto sub_tmp = ring_add(ring_add(a9, a10), a11);
  auto grad = ring_sub(add_tmp, sub_tmp);
 
  if (comm->getRank() == 0) {
    auto a12 = ring_arshift(ring_mmul(iden2, x_r1, N, M, K), bits);
    auto a13 = ring_arshift(ring_mmul(ring_mmul(w_r2, x_r1T, N, K, M), x_r1, N, M, K), bits);
    auto a14 = ring_arshift(ring_mul(four, ring_mmul(y_r3T, x_r1, N, M, K)), bits);
    // auto a12 = ring_arshift(ring_mmul(ring_arshift(ring_mmul(w_r2, x_r1T, K, M, N), bits), x_r1, K, N, M), bits);
    // auto a13 = ring_arshift(ring_mul(four, ring_arshift(ring_mmul(y_r3T, x_r1, K, N, M), bits)), bits);
    auto tmp = ring_sub(ring_add(a11, a12), a14);
    ring_add_(grad, tmp);
  }

  //1106,1107
  // size_t bits = 18;
  // auto iden2 = ring_add(ring_ones(field, K * M), ring_ones(field, K * M));
  // auto iden4 = ring_add(iden2, iden2);
  // auto iden8 = ring_add(iden4, iden4);
  // auto wxT = ring_arshift(ring_mmul(w_r2, x_r1T, K, M, N), bits);
  // // auto yT = ring_arshift(ring_mul(iden8, y_r3T), bits);
  // auto yT = ring_mul(iden8, y_r3T);
  // auto grad = ring_arshift(ring_mmul(ring_sub(ring_add(iden4, wxT), yT), x_r1, K, N, M), bits);


  // std::cout<<"----grad----"<<std::endl;
  // for(int64_t n = 0; n < grad.numel(); n++) {
  //   std::cout<<grad.at<int32_t>(n)<<std::endl;
  // }
  return grad.as(w.eltype());
}


}  // namespace spu::mpc::semi2k