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

// clang-format off
// To run the example, start two terminals:
// > bazel run //examples/cpp:simple_lr -- -rank 0 -dataset examples/data/breast_cancer_b.csv -has_label=true
// > bazel run //examples/cpp:simple_lr -- -rank 1 -dataset examples/data/breast_cancer_a.csv
// clang-format on

// > bazel run //examples/cpp:simple_lr -- --dataset=examples/data/heart_b.csv --has_label=true
// > bazel run //examples/cpp:simple_lr -- --dataset=examples/data/heart_a.csv --rank=1

#include <fstream>
#include <iostream>
#include <vector>

#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "spu/device/io.h"
#include "spu/hal/hal.h"
#include "spu/hal/test_util.h"
#include "spu/hal/type_cast.h"
#include "float.h"
#include <time.h>

clock_t start1, end1, start2, end2;
// clock_t ss_start, ss_end;



spu::hal::Value train_step(spu::HalContext* ctx, const spu::hal::Value& x,
                           const spu::hal::Value& y, const spu::hal::Value& w) {
  // Padding x
  auto padding = spu::hal::constant(ctx, 1.0F, {x.shape()[0], 1});
  // std::cout<<""<<std::endl;
  auto padded_x =
      spu::hal::concatenate(ctx, {x, spu::hal::p2s(ctx, padding)}, 1);

  std::cout<<"------The first matmul------"<<std::endl;
  // xt::xarray<float> revealed_x = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, x)); 
  // xt::xarray<float> revealed_w = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, w));
  // xt::xarray<float> revealed_y = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, y));
  // xt::xarray<float> revealed_padded_x = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, padded_x));

  // std::cout<<"*****revealed_x.shape(0), revealed_x.shape(1)*****"<<std::endl;
  // std::cout<<revealed_x.shape(0)<<std::endl;
  // std::cout<<revealed_x.shape(1)<<std::endl;
  // std::cout<<"*****revealed_w.shape(0), revealed_w.shape(1)*****"<<std::endl;
  // std::cout<<revealed_w.shape(0)<<std::endl;
  // std::cout<<revealed_w.shape(1)<<std::endl;
  // std::cout<<"*****revealed_y.shape(0), revealed_y.shape(1)*****"<<std::endl;
  // std::cout<<revealed_y.shape(0)<<std::endl;
  // std::cout<<revealed_y.shape(1)<<std::endl;
  // std::cout<<"*****revealed_padded_x.shape(0), revealed_padded_x.shape(1)*****"<<std::endl;
  // std::cout<<revealed_padded_x.shape(0)<<std::endl;
  // std::cout<<revealed_padded_x.shape(1)<<std::endl;

  auto mmul = spu::hal::matmul(ctx, padded_x, w);
  // auto mmul = spu::hal::matmul(ctx, x, w);
  start2 = clock();
  auto pred = spu::hal::logistic(ctx, mmul);
  end2 = clock();

  SPDLOG_DEBUG("[SSLR] Err = Pred - Y");
  auto err = spu::hal::sub(ctx, pred, y);

  // std::cout<<"--------------original err-------------"<<std::endl;

  // xt::xarray<float> revealed_err = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, err));

  // for(size_t i = 0; i < revealed_err.shape(0); i++) {
  //   for(size_t j = 0; j < revealed_err.shape(1); j++) {
  //     std::cout<<revealed_err(i, j)<<std::endl;
  //   }
  // }

  
  SPDLOG_DEBUG("[SSLR] Grad = X.t * Err");
  std::cout<<"------The second matmul------"<<std::endl;
  //   xt::xarray<float> revealed_x1 = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, spu::hal::transpose(ctx, padded_x)));
  // xt::xarray<float> revealed_err = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, err));
  // std::cout<<"*****revealed_x1.shape(0), revealed_x1.shape(1)*****"<<std::endl;
  // std::cout<<revealed_x1.shape(0)<<std::endl;
  // std::cout<<revealed_x1.shape(1)<<std::endl;
  // std::cout<<"*****revealed_err.shape(0), revealed_err.shape(1)*****"<<std::endl;
  // std::cout<<revealed_err.shape(0)<<std::endl;
  // std::cout<<revealed_err.shape(1)<<std::endl;
  auto grad = spu::hal::matmul(ctx, spu::hal::transpose(ctx, padded_x), err);

  SPDLOG_DEBUG("[SSLR] Step = LR / B * Grad");
  auto lr = spu::hal::constant(ctx, 0.5F);
  auto msize = spu::hal::constant(ctx, static_cast<float>(y.shape()[0]));
  // std::cout<<"------The third mul------"<<std::endl;
  auto p1 = spu::hal::mul(ctx, lr, spu::hal::reciprocal(ctx, msize));

  // std::cout<<"------The fourth mul------"<<std::endl;
  auto step =
      spu::hal::mul(ctx, spu::hal::broadcast_to(ctx, p1, grad.shape()), grad);

  // std::cout<<"----------revealed step-----------"<<std::endl;
  // xt::xarray<float> revealed_step = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, step));

  // for(size_t i = 0; i < revealed_step.shape(0); i++) {
  //   for(size_t j = 0; j < revealed_step.shape(1); j++) {
  //     std::cout<<revealed_step(i, j)<<std::endl;
  //   }
  // }

  SPDLOG_DEBUG("[SSLR] W = W - Step");
  auto new_w = spu::hal::sub(ctx, w, step);
  // xt::xarray<float> revealed_new_w = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, new_w));
  // std::cout<<"*****revealed_new_w.shape(0), revealed_new_w.shape(1)*****"<<std::endl;
  // std::cout<<revealed_new_w.shape(0)<<std::endl;
  // std::cout<<revealed_new_w.shape(1)<<std::endl;

  return new_w;
}

spu::hal::Value train(spu::HalContext* ctx, const spu::hal::Value& x,
                      const spu::hal::Value& y, size_t num_epoch,
                      size_t bsize) {

  const size_t num_iter = x.shape()[0] / bsize;
  auto w = spu::hal::constant(ctx, 0.0F, {x.shape()[1] + 1, 1});
  // std::cout<<"x.shape()[0] : "<<x.shape()[0]<<std::endl;
  // std::cout<<"x.shape()[1] : "<<x.shape()[1]<<std::endl;
  // std::cout<<"w.shape()[0] : "<<w.shape()[0]<<std::endl;
  // std::cout<<"w.shape()[1] : "<<w.shape()[1]<<std::endl;
  // std::cout<<"y.shape()[0] : "<<y.shape()[0]<<std::endl;
  // std::cout<<"y.shape()[0] : "<<y.shape()[1]<<std::endl;
  // std::cout<<"num_epoch : "<<num_epoch<<std::endl;
  // std::cout<<"num_iter : "<<num_iter<<std::endl;
  //1110
  // w = train_step(ctx, x, y, w);
  // Run train loop
  for (size_t epoch = 0; epoch < num_epoch; ++epoch) {
    for (size_t iter = 0; iter < num_iter; ++iter) {
      SPDLOG_INFO("Running train iteration {}", iter);

      const int64_t rows_beg = iter * bsize;
      const int64_t rows_end = rows_beg + bsize;

      const auto x_slice =
          spu::hal::slice(ctx, x, {rows_beg, 0}, {rows_end, x.shape()[1]}, {});
    
      std::cout<<"rows_beg : "<<rows_beg<<std::endl;
      std::cout<<"rows_end : "<<rows_end<<std::endl;

      const auto y_slice =
          spu::hal::slice(ctx, y, {rows_beg, 0}, {rows_end, y.shape()[1]}, {});

      w = train_step(ctx, x_slice, y_slice, w);

    }
  }

  return w;
}

spu::hal::Value inference(spu::HalContext* ctx, const spu::hal::Value& x,
                          const spu::hal::Value& weight) {
  auto padding = spu::hal::constant(ctx, 1.0F, {x.shape()[0], 1});
  auto padded_x =
      spu::hal::concatenate(ctx, {x, spu::hal::p2s(ctx, padding)}, 1);
  std::cout<<"------The fifth matmul------"<<std::endl;
  return spu::hal::matmul(ctx, padded_x, weight);
}

float SSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  float sse = 0;

  for (auto y_true_iter = y_true.begin(), y_pred_iter = y_pred.begin();
       y_true_iter != y_true.end() && y_pred_iter != y_pred.end();
       ++y_pred_iter, ++y_true_iter) {
    sse += std::pow(*y_true_iter - *y_pred_iter, 2);
  }
  return sse;
}

float MSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  auto sse = SSE(y_true, y_pred);

  return sse / static_cast<float>(y_true.size());
}

//lj
float AUC(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  size_t i, j;
  float auc = 0;
  size_t cnt1 = 0, cnt2 = 0;
  std::vector<size_t> pos;
  std::vector<size_t> neg;
  for(i = 0; i < y_true.shape(0); i++) {
    if(y_true(i, 0) == 1) {
      pos.push_back(i);
    }
    else if(y_true(i, 0) == 0) {
      neg.push_back(i);
    }
  }

  for(i = 0; i < pos.size(); i++) {
    for(j = 0; j < neg.size(); j++) {
      if(y_pred(pos[i], 0) > y_pred(neg[j], 0)) {
        auc += 1;
        cnt1 += 1;
      }else if(y_pred(pos[i], 0) == y_pred(neg[j], 0)) {
        auc += 0.5;
        cnt2 += 1;
      }
    }
  }
  return auc / (pos.size() * neg.size());

}


llvm::cl::opt<std::string> Dataset("dataset", llvm::cl::init("data.csv"),
                                   llvm::cl::desc("only csv is supported"));
llvm::cl::opt<uint32_t> SkipRows(
    "skip_rows", llvm::cl::init(1),
    llvm::cl::desc("skip number of rows from dataset"));
llvm::cl::opt<bool> HasLabel(
    "has_label", llvm::cl::init(false),
    llvm::cl::desc("if true, label is the last column of dataset"));
llvm::cl::opt<uint32_t> BatchSize("batch_size", llvm::cl::init(128),
                                  llvm::cl::desc("size of each batch"));
llvm::cl::opt<uint32_t> NumEpoch("num_epoch", llvm::cl::init(1),
                                 llvm::cl::desc("number of epoch"));

std::pair<spu::hal::Value, spu::hal::Value> infeed(spu::HalContext* hctx,
                                                   const xt::xarray<float>& ds,
                                                   bool self_has_label) {
  spu::device::ColocatedIo cio(hctx);
  if (self_has_label) {
    // the last column is label.
    using namespace xt::placeholders;  // required for `_` to work
    xt::xarray<float> dx =
        xt::view(ds, xt::all(), xt::range(_, ds.shape(1) - 1));
    xt::xarray<float> dy =
        xt::view(ds, xt::all(), xt::range(ds.shape(1) - 1, _));
    cio.hostSetVar(fmt::format("x-{}", hctx->lctx()->Rank()), dx);
    cio.hostSetVar("label", dy);
  } else {
    cio.hostSetVar(fmt::format("x-{}", hctx->lctx()->Rank()), ds);
  }
  cio.sync();

  auto x = cio.deviceGetVar("x-0");
  // Concatnate all slices
  for (size_t idx = 1; idx < cio.getWorldSize(); ++idx) {
    x = spu::hal::concatenate(
        hctx, {x, cio.deviceGetVar(fmt::format("x-{}", idx))}, 1);
  }
  auto y = cio.deviceGetVar("label");

  return std::make_pair(x, y);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  // read dataset.
  xt::xarray<float> ds;
  {
    std::ifstream file(Dataset.getValue());
    if (!file) {
      spdlog::error("open file={} failed", Dataset.getValue());
      exit(-1);
    }
    ds = xt::load_csv<float>(file, ',', SkipRows.getValue());
  }

  // std::cout<<"loading ds.shape(0) : "<<ds.shape(0)<<std::endl;
  // std::cout<<"loading ds.shape(1) : "<<ds.shape(1)<<std::endl;

  auto hctx = MakeHalContext();

  const auto& [x, y] = infeed(hctx.get(), ds, HasLabel.getValue());

  start1 = clock();

  const auto w =
      train(hctx.get(), x, y, NumEpoch.getValue(), BatchSize.getValue());
  
  end1 = clock();

  // const auto scores = inference(hctx.get(), x, w);


  // xt::xarray<float> revealed_labels = spu::hal::test::dump_public_as<float>(
  //     hctx.get(), spu::hal::reveal(hctx.get(), y));
  // xt::xarray<float> revealed_scores = spu::hal::test::dump_public_as<float>(
  //     hctx.get(), spu::hal::reveal(hctx.get(), scores));
      
  // auto mse = MSE(revealed_labels, revealed_scores);
  // auto auc = AUC(revealed_labels, revealed_scores);
  // std::cout << "MSE = " << mse << "\n";
  // std::cout << "AUC = " << auc << "\n"; 
  std::cout << "total time = "<<double(end1 - start1)/CLOCKS_PER_SEC<<"s"<<std::endl; 
  // std::cout << "logistic time = "<<double(end2 - start2)/CLOCKS_PER_SEC<<"s"<<std::endl; 
  return 0;
}