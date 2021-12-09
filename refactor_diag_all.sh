#!/bin/bash

# Devign
echo ***** Devign
python data_processing/refactor_reveal.py --mode diag -i data/devign -o out/refactored_devign_seed1 || exit 1

echo ***** Devign + Threshold0.5
python data_processing/refactor_reveal.py --mode diag -i data/devign -o out/refactored_devign_threshold0.5 || exit 1

echo ***** Devign + NoNewName
python data_processing/refactor_reveal.py --mode diag -i data/devign -o out/refactored_devign_no_new_name || exit 1

echo ***** Devign + NoNewName + Threshold0.5
python data_processing/refactor_reveal.py --mode diag -i data/devign -o out/refactored_devign_noname_threshold0.5 || exit 1

echo ***** Devign + NoNewName + Threshold0.75
python data_processing/refactor_reveal.py --mode diag -i data/devign -o out/refactored_devign_noname_threshold0.75 || exit 1


# Reveal
echo ***** Reveal
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_chrome_debian_seed0 || exit 1

echo ***** Reveal + BuggyOnly
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed0 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed1 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed2 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed3 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed4 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed5 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed6 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed7 || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_buggyonly_seed8 || exit 1

echo ***** Reveal + NoName + Threshold0.75
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75 || exit 1

echo ***** Reveal + NoName + Threshold0.75 + BuggyOnly
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-0_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-1_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-2_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-3_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-4_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-5_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-6_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-7_buggyonly || exit 1
python data_processing/refactor_reveal.py --mode diag -i data/chrome_debian -o out/refactored_reveal_noname_threshold0.75_seed-8_buggyonly || exit 1

#refactored_buggyonly_devign_seed0       refactored_devign_seed1
#refactored_buggyonly_seed0              refactored_devign_threshold0.5
#refactored_buggyonly_seed1              refactored_noname_threshold0.5
#refactored_buggyonly_seed2              refactored_reveal_noname_threshold0.75
#refactored_buggyonly_seed3              refactored_reveal_noname_threshold0.75_seed-0_buggyonly
#refactored_buggyonly_seed4              refactored_reveal_noname_threshold0.75_seed-1_buggyonly
#refactored_buggyonly_seed5              refactored_reveal_noname_threshold0.75_seed-2_buggyonly
#refactored_buggyonly_seed6              refactored_reveal_noname_threshold0.75_seed-3_buggyonly
#refactored_buggyonly_seed7              refactored_reveal_noname_threshold0.75_seed-4_buggyonly
#refactored_buggyonly_seed8              refactored_reveal_noname_threshold0.75_seed-5_buggyonly
#refactored_chrome_debian_seed0          refactored_reveal_noname_threshold0.75_seed-6_buggyonly
#refactored_chrome_debian_seed1          refactored_reveal_noname_threshold0.75_seed-7_buggyonly
#refactored_devign_no_new_name           refactored_reveal_noname_threshold0.75_seed-8_buggyonly
#refactored_devign_noname_threshold0.5   refactored_reveal_noname_threshold0.75_seed-9_buggyonly
#refactored_devign_noname_threshold0.75
