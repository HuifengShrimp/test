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

#include "spu/mpc/beaver/beaver_tfp.h"

#include <random>

#include "yasl/link/link.h"
#include "yasl/utils/serialize.h"

#include "spu/mpc/beaver/prg_tensor.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {
namespace {

uint128_t GetHardwareRandom128() {
  std::random_device rd;
  // call random_device four times, make sure uint128 is random in 2^128 set.
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return yasl::MakeUint128(lhs, rhs);
}

}  // namespace

BeaverTfpUnsafe::BeaverTfpUnsafe(std::shared_ptr<yasl::link::Context> lctx)
    : lctx_(lctx), seed_(GetHardwareRandom128()), counter_(0) {
  auto buf = yasl::SerializeUint128(seed_);
  std::vector<yasl::Buffer> all_bufs =
      yasl::link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = yasl::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, lctx_->WorldSize(), seed);
    }
  }
}

Beaver::Triple BeaverTfpUnsafe::Mul(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustMul(descs);
  }

  return {a, b, c};
}

Beaver::Triple BeaverTfpUnsafe::Dot(FieldType field, size_t M, size_t N,
                                    size_t K) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, M * K, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, K * N, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, M * N, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustDot(descs, M, N, K);
  }

  return {a, b, c};
}

Beaver::Triple BeaverTfpUnsafe::And(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustAnd(descs);
  }

  return {a, b, c};
}

Beaver::Pair BeaverTfpUnsafe::Trunc(FieldType field, size_t size, size_t bits) {
  std::vector<PrgArrayDesc> descs(2);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == 0) {
    b = tp_.adjustTrunc(descs, bits);
  }

  return {a, b};
}

ArrayRef BeaverTfpUnsafe::RandBit(FieldType field, size_t size) {
  PrgArrayDesc desc{};
  auto a = prgCreateArray(field, size, seed_, &counter_, &desc);

  if (lctx_->Rank() == 0) {
    a = tp_.adjustRandBit(desc);
  }

  return a;
}

static void TransposeInplace(ArrayRef mat, size_t nrows, size_t ncols) {
  YASL_ENFORCE_EQ((size_t)mat.numel(), nrows * ncols);
  const auto field = mat.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto xmat = xt_mutable_adapt<ring2k_t>(mat);
    xmat.reshape({nrows, ncols});
    auto xmatT = xt::eval(xt::transpose(xmat));
    //1027
    std::copy_n(xmatT.begin(), xmatT.size(), xmat.data());
  });
}


//lj
Beaver::Lr_set BeaverTfpUnsafe::lr(FieldType field, size_t M, size_t K, size_t N) {

    //1103 - zero sharing
    // auto a = ring_zeros(field, M * N);
    // auto b = ring_zeros(field, K * N);
    // auto c = ring_zeros(field, M * K);
    // auto d = ring_zeros(field, K * M);
    // auto f = ring_zeros(field, K * N);
    // auto g = ring_zeros(field, K * N);
    // auto h = ring_zeros(field, N * N);

    auto r1 = ring_zeros(field, K * M);
    auto r2 = ring_zeros(field, N * M);
    auto r3 = ring_zeros(field, K * N);

    auto r1T = r1.clone();
    TransposeInplace(r1T, K, M);
    auto r3T = r3.clone();
    TransposeInplace(r3T, K, N);

    auto c1 = ring_mmul(r2, r1T, N, K, M);
    auto c3 = ring_mmul(c1, r1, N, M, K);
    auto c4 = ring_mmul(r3T, r1, N, M, K);
    auto c5 = ring_mmul(r1T, r1, M, M, K);
    
    // if (lctx_->Rank() == 0) {
    //   r1 = ring_sub(a, r1);
    //   r2 = ring_sub(b, r2);
    //   r3 = ring_sub(c, r3);
    //   c1 = ring_sub(d, c1);
    //   c3 = ring_sub(f, c3);
    //   c4 = ring_sub(g, c4);
    //   c5 = ring_sub(h, c5);
    // }
    return {r1, r2, r3, c1, c3, c4, c5};
}

Beaver::Triple BeaverTfpUnsafe::LR_TEST(FieldType field, size_t M, size_t N, size_t K) {
  auto r1 = ring_zeros(field, M * N);
  auto r2 = ring_zeros(field, K * N);
  auto r3 = ring_zeros(field, M * K);

  // auto a = ring_ones(field, M * N);
  // auto b = ring_ones(field, K * N);
  // auto c = ring_ones(field, M * K);

  // if(lctx_->Rank() == 0) {
  //   a = ring_sub(r1, a);
  //   b = ring_sub(r2, b);
  //   c = ring_sub(r3, c);
  // }

  // return {a, b, c};
  return {r1, r2, r3}; 
}

}  // namespace spu::mpc