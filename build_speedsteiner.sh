#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p work/bin
"${CXX:-c++}" -std=c++20 -O3 -DNDEBUG -DFMT_HEADER_ONLY -DFMT_CONSTEVAL= \
  -DBOOST_NO_CXX98_FUNCTION_BASE \
  -I work/vendor/directed-steiner-tree-code-master/dst/include \
  -I work/vendor/fmt-10.2.1/include -I work/vendor/usr/include \
  speedsteiner_adapter.cpp -o work/bin/speedsteiner_adapter
