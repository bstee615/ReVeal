for i in $(seq 0 4); do sbatch -J "reveal_basic_$i" batch.sh bash train_reveal.sh basic_plus_refactored_cross_validated/fold_$i/; done
