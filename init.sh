#!/bin/bash
# Call with "conda activate pytorch_latest_p37; bash init.sh 1dsiQYjIdtS5JRyiHDfrurzhZvdOF8_-E 2>&1 | tee all.log"
# To fix the reveal problem: "(cd thesis && sed -i.back 's/python/python -u/g' train_reveal_end.sh && bash train_reveal_end.sh ~/data 2>&1 | tee ~/reveal-end.log)"
# Zip up the results: "tar zvcf save.tgz data/after_ggnn/*.json data/models/ thesis *.log"


set -x

url="$1"

if [ -z "$url" ]
then
  echo "Error"
  exit 1
fi

echo "$url"

cd ~ || exit 1

#conda env list
#conda activate pytorch_latest_p37

pip install gdown
mkdir -p ~/data/ggnn_input
#Example: gdown https://drive.google.com/uc?id=1KLqZx5a5fdTawlkUKI3wELY9JUTvE2FH
#Link to thesis: https://drive.google.com/file/d/1lHKX5nQvUJPUfFanV6YzvSbgQW7FGZaU/view?usp=sharing
start_url="https://drive.google.com/uc?id=1lHKX5nQvUJPUfFanV6YzvSbgQW7FGZaU"
if [ ! -f ~/thesis.tar.gz ]
then
  gdown "$start_url" -O ~/thesis.tar.gz || exit 1
  tar zxf ~/thesis.tar.gz || exit 1
fi
if [ ! -f ~/data/ggnn_input/processed.bin ]
then
  gdown "https://drive.google.com/uc?id=$url" -O ~/data/ggnn_input/processed.bin || exit 1
fi

cd ~/thesis || exit 1
pip install -r requirements.txt
pip install -r Vuld_SySe/requirements.txt
pip install -r Vuld_SySe/representation_learning/requirements.txt

bash "train_reveal.sh" ~/data 2>&1 | tee ~/reveal.log
bash "train_devign.sh" ~/data 2>&1 | tee ~/devign.log
