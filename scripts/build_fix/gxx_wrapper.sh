#!/bin/bash
# Wrapper for g++ that replaces -fopenmp-simd with -fopenmp (GCC 4.8 doesn't support -fopenmp-simd).
# Use gcc for C++ if g++ is not installed.
args=()
for a in "$@"; do
  if [[ "$a" == "-fopenmp-simd" ]]; then
    args+=("-fopenmp")
  else
    args+=("$a")
  fi
done
if [[ -x /usr/bin/g++ ]]; then
  exec /usr/bin/g++ "${args[@]}"
else
  exec /usr/bin/gcc -lstdc++ "${args[@]}"
fi
