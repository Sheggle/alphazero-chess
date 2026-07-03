#!/bin/bash
source /venv/main/bin/activate
cd /root/research
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
{
  echo "===ENCODE==="
  python -u bench_encode.py --no-corr
  echo "===SELFPLAY (back-to-back fast vs oldfast)==="
  echo "#### FAST 1 ####";    python -u bench_selfplay.py 1 25
  echo "#### OLDFAST 1 ####"; python -u bench_selfplay.py 1 25 --oldfast
  echo "#### FAST 8 ####";    python -u bench_selfplay.py 8 25
  echo "#### OLDFAST 8 ####"; python -u bench_selfplay.py 8 25 --oldfast
  echo ALLDONE2
} > /root/sp5.out 2>&1
