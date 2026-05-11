for benchmark in val test; do
    python test.py --benchmark=$benchmark --verbose
    python eval_hku.py --benchmark=$benchmark --method=lgr
done
