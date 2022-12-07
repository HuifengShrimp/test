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
// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/perfect_logit_a.csv --has_label=true
// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/perfect_logit_b.csv --rank=1
// clang-format on

// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/breast_cancer_b.csv --has_label=true
// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/breast_cancer_a.csv --rank=1

// > bazel run //examples/cpp:fss_lr -- -rank 0 -dataset examples/data/breast_cancer_b.csv -has_label=true
// > bazel run //examples/cpp:fss_lr -- -rank 1 -dataset examples/data/breast_cancer_a.csv

// > bazel test //spu/mpc/beaver:beaver_test

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
#include <time.h>

//lj
#include "float.h"

clock_t fss_start, fss_end;

spu::hal::Value train_step(spu::HalContext* ctx, const spu::hal::Value& x,
                           const spu::hal::Value& y, const spu::hal::Value& w) {

  // std::cout<<"-----train_step----"<<std::endl;

  // Padding x
  auto padding = spu::hal::constant(ctx, 1.0F, {x.shape()[0], 1});
  auto padded_x =
      spu::hal::concatenate(ctx, {x, spu::hal::p2s(ctx, padding)}, 1);

  // std::cout<<"--------------w-------------"<<std::endl;

  // xt::xarray<float> revealed_y = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, w));

  // for(size_t i = 0; i < revealed_y.shape(0); i++) {
  //   for(size_t j = 0; j < revealed_y.shape(1); j++) {
  //     std::cout<<revealed_y(i, j)<<std::endl;
  //   }
  // }

  auto grad = spu::hal::logreg(ctx, padded_x, w, y);
  // std::cout<<"----logreg finish----"<<std::endl;

  // xt::xarray<float> revealed_grad = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, grad));

  // for(size_t i = 0; i < revealed_grad.shape(0); i++) {
  //   for(size_t j = 0; j < revealed_grad.shape(1); j++) {
  //     std::cout<<revealed_grad(i, j)<<std::endl;
  //   }
  // }

  auto lr = spu::hal::constant(ctx, 0.5F);
  auto four = spu::hal::constant(ctx, static_cast<float>(4.0F));
  auto msize = spu::hal::constant(ctx, static_cast<float>(y.shape()[0]));
  auto msize_ = spu::hal::mul(ctx, four, msize);
  auto p1 = spu::hal::mul(ctx, lr, spu::hal::reciprocal(ctx, msize_));
  auto step = spu::hal::mul(ctx, spu::hal::broadcast_to(ctx, p1, grad.shape()), grad);
  auto step1 = spu::hal::reshape(ctx, step, w.shape());
  // std::cout<<"----grad.shape0 : "<<grad.shape()[0]<<std::endl;
  // std::cout<<"----grad.shape1 : "<<grad.shape()[1]<<std::endl;


  
  // // std::cout<<"------The first mul------"<<std::endl;
  // auto msize = spu::hal::mul(ctx, spu::hal::constant(ctx, 4.0F), spu::hal::constant(ctx, static_cast<float>(y.shape()[0])));

  // // std::cout<<"------The second mul------"<<std::endl;
  // auto p1 = spu::hal::mul(ctx, lr, spu::hal::reciprocal(ctx, msize));

  // // std::cout<<"------The third mul------"<<std::endl;
  // auto step =
  //      spu::hal::mul(ctx, spu::hal::broadcast_to(ctx, p1, grad.shape()), grad);

  // xt::xarray<float> revealed_step = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, step));

  // for(size_t i = 0; i < revealed_step.shape(0); i++) {
  //   for(size_t j = 0; j < revealed_step.shape(1); j++) {
  //     std::cout<<revealed_step(i, j)<<std::endl;
  //   }
  // }

  // std::cout<<"------return grad------"<<std::endl;
  // xt::xarray<float> revealed_grad = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, step));

  // for(size_t i = 0; i < revealed_grad.shape(0); i++) {
  //   for(size_t j = 0; j < revealed_grad.shape(1); j++) {
  //     std::cout<<revealed_grad(i, j)<<std::endl;
  //   }
  // }

  // SPDLOG_DEBUG("[SSLR] Step = LR / B * Grad");
  // auto lr = spu::hal::constant(ctx, 0.5F);
  // auto msize = spu::hal::constant(ctx, static_cast<float>(y.shape()[0]));
  // // std::cout<<"------The third mul------"<<std::endl;
  // auto p1 = spu::hal::mul(ctx, lr, spu::hal::reciprocal(ctx, msize));

  // // std::cout<<"------The fourth mul------"<<std::endl;
  // auto step =
  //     spu::hal::mul(ctx, spu::hal::broadcast_to(ctx, p1, grad.shape()), grad);

  // std::cout<<"------return step------"<<std::endl;
  // xt::xarray<float> revealed_step = spu::hal::test::dump_public_as<float>(
  //     ctx, spu::hal::reveal(ctx, step));

  // for(size_t i = 0; i < revealed_step.shape(0); i++) {
  //   for(size_t j = 0; j < revealed_step.shape(1); j++) {
  //     std::cout<<revealed_step(i, j)<<std::endl;
  //   }
  // }

  SPDLOG_DEBUG("[FSS-LR] W = W - Step");
  auto new_w = spu::hal::sub(ctx, w, step1);
  // std::cout<<"new w finish"<<std::endl;
  return new_w;
}

spu::hal::Value train(spu::HalContext* ctx, const spu::hal::Value& x,
                      const spu::hal::Value& y, size_t num_epoch,
                      size_t bsize) {
  const size_t num_iter = x.shape()[0] / bsize;
  auto w = spu::hal::constant(ctx, 0.0F, {x.shape()[1] +1, 1});
  //1110-lj
  // w = train_step(ctx, x, y, w);
  // std::cout<<"x.shape()[0] : "<<x.shape()[0]<<std::endl;
  // std::cout<<"x.shape()[1] : "<<x.shape()[1]<<std::endl;
  // std::cout<<"w.shape()[0] : "<<w.shape()[0]<<std::endl;
  // std::cout<<"w.shape()[1] : "<<w.shape()[1]<<std::endl;
  // std::cout<<"y.shape()[0] : "<<y.shape()[0]<<std::endl;
  // std::cout<<"y.shape()[0] : "<<y.shape()[1]<<std::endl;
  // std::cout<<"num_epoch : "<<num_epoch<<std::endl;
  // std::cout<<"num_iter : "<<num_iter<<std::endl;
  // Run train loop
  for (size_t epoch = 0; epoch < num_epoch; ++epoch) {
    for (size_t iter = 0; iter < num_iter; ++iter) {
      SPDLOG_INFO("Running train iteration {}", iter);

      const int64_t rows_beg = iter * bsize;
      const int64_t rows_end = rows_beg + bsize;

      const auto x_slice =
          spu::hal::slice(ctx, x, {rows_beg, 0}, {rows_end, x.shape()[1]}, {});

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
  // std::cout<<"------The fourth mul------"<<std::endl;
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
      }else if(y_pred(pos[i], 0) == y_pred(neg[j], 0)) {
        auc += 0.5;
      }
    }
  }

  return auc / (pos.size() * neg.size());

}

//lj
// float multi_AUC(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  
// }

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

  std::cout<<"loading ds.shape(0) : "<<ds.shape(0)<<std::endl;
  std::cout<<"loading ds.shape(1) : "<<ds.shape(1)<<std::endl;

  //lj
  // size_t i, j;
  // double MAX_ELEMENT = -DBL_MAX;
  // double MIN_ELEMENT = DBL_MAX;
  // //find the maximum, minimum
  // if(!HasLabel.getValue()){
  //     for(i = 0; i < ds.shape(0); i++){
  //       for(j = 0; j < ds.shape(1); j++){
  //         if(ds(i, j) > MAX_ELEMENT){
  //           MAX_ELEMENT = ds(i, j);
  //         }
  //         if(ds(i, j) < MIN_ELEMENT){
  //           MIN_ELEMENT = ds(i, j);
  //         }
  //       }
  //     }
  // }else{
  //   for(i = 0; i < ds.shape(0); i++){
  //       for(j = 0; j < ds.shape(1)-1; j++){
  //         if(ds(i, j) > MAX_ELEMENT){
  //           MAX_ELEMENT = ds(i, j);
  //         }
  //         if(ds(i, j) < MIN_ELEMENT){
  //           MIN_ELEMENT = ds(i, j);
  //         }
  //       }
  //     }
  // }

  // double delta = MAX_ELEMENT - MIN_ELEMENT;
  // //normalization
  // if(!HasLabel.getValue()){
  //     for(i = 0; i < ds.shape(0); i++){
  //       for(j = 0; j < ds.shape(1); j++){
  //         ds(i, j) = (ds(i, j)-MIN_ELEMENT)/delta;
  //       }
  //     }
  // }else{
  //   for(i = 0; i < ds.shape(0); i++){
  //       for(j = 0; j < ds.shape(1)-1; j++){
  //         ds(i, j) = (ds(i, j)-MIN_ELEMENT)/delta;
  //       }
  //     }
  // }

  auto hctx = MakeHalContext();

  const auto& [x, y] = infeed(hctx.get(), ds, HasLabel.getValue());
  
  fss_start = clock();
  const auto w =
      train(hctx.get(), x, y, NumEpoch.getValue(), BatchSize.getValue());
  fss_end = clock();

  // const auto scores = inference(hctx.get(), x, w);

  // xt::xarray<float> revealed_labels = spu::hal::test::dump_public_as<float>(
  //     hctx.get(), spu::hal::reveal(hctx.get(), y));
  // xt::xarray<float> revealed_scores = spu::hal::test::dump_public_as<float>(
  //     hctx.get(), spu::hal::reveal(hctx.get(), scores));

  // // // for(i = 0; i < revealed_scores.shape(0); i++){
  // // //   for(j = 0; j < revealed_scores.shape(1); j++){
  // // //     std::cout<<revealed_scores(i, j)<<std::endl;
  // // //   }
  // // // }

  // auto mse = MSE(revealed_labels, revealed_scores);
  // auto auc = AUC(revealed_labels, revealed_scores);

  // std::cout << "MSE = " << mse << "\n";
  // std::cout << "AUC = " << auc << "\n"; 
  std::cout << "total time = "<<double(fss_end - fss_start)/CLOCKS_PER_SEC<<"s"<<std::endl; 

  return 0;
}