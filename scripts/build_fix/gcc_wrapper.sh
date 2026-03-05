#!/bin/bash
# Wrapper for gcc: (1) -fopenmp-simd -> -fopenmp for GCC 4.8; (2) force C99 for spacy/cython C code.
args=()
has_std=0
for a in "$@"; do
  if [[ "$a" == "-fopenmp-simd" ]]; then
    args+=("-fopenmp")
  elif [[ "$a" == -std=* ]]; then
    args+=("$a")
    has_std=1
  else
    args+=("$a")
  fi
done
[[ $has_std -eq 0 ]] && args=("-std=gnu99" "${args[@]}")
exec /usr/bin/gcc "${args[@]}"
