#!/usr/bin/env bash
# Builds PipeANN-Filter (pipeann.patch on the pinned commit) and Faiss-SSD (faiss_ssd.cpp on Faiss v1.15.0) under BASELINES
# (default baselines/build). A finished build is kept, and Faiss-SSD is compiled again when faiss_ssd.cpp changes.
#
# Needs: build-essential cmake libboost-all-dev libeigen3-dev libjemalloc-dev libgoogle-perftools-dev libaio-dev liburing-dev
# libmkl-dev intel-mkl
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${BASELINES:-$here/build}
jobs=$(nproc)
mkdir -p "$out"
cd "$out"
out=$(pwd)

PIPEANN_COMMIT=6c7d9dae21bc0590aa570c8d2998f34abf37f61c
FAISS_COMMIT=20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed

if [ ! -x "$out/PipeANN/build/tests/search_disk_index_filtered" ]; then
  [ -d PipeANN ] || git clone -q --recurse-submodules https://github.com/thustorage/PipeANN.git
  cd PipeANN
  git checkout -q "$PIPEANN_COMMIT"
  git submodule update -q --init --recursive
  git diff --quiet -- tests/search_disk_index_filtered.cpp && git apply "$here/pipeann.patch"
  (cd third_party/liburing && ./configure > /dev/null)
  cmake -B build . -DCMAKE_BUILD_TYPE=Release > /dev/null
  make -C build -j"$jobs" build_disk_index_filtered search_disk_index_filtered > /dev/null
  cd "$out"
fi

if [ ! -e "$out/faiss/build/faiss/libfaiss_avx2.so" ]; then
  [ -d faiss ] || git clone -q https://github.com/facebookresearch/faiss.git
  cd faiss
  git checkout -q "$FAISS_COMMIT"
  cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 \
    -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DBLA_VENDOR=Intel10_64lp > /dev/null
  make -C build -j"$jobs" faiss_avx2 > /dev/null
  cd "$out"
fi
if [ ! -x "$out/faiss_ssd" ] || [ "$here/faiss_ssd.cpp" -nt "$out/faiss_ssd" ]; then
  g++ -O3 -march=native -std=c++17 -fopenmp -Wall -I"$out/faiss" "$here/faiss_ssd.cpp" -o faiss_ssd \
    -L"$out/faiss/build/faiss" -lfaiss_avx2 -Wl,-rpath,"$out/faiss/build/faiss" -luring -lpthread
fi
