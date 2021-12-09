sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get -y update
sudo apt-get install -y gcc-4.9
sudo apt-get upgrade -y libstdc++6
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX

(cd thesis && sed -i.back 's/python/python -u/g' train_reveal_end.sh && bash train_reveal_end.sh ~/data 2>&1 | tee ~/reveal-end.log)

#cp reveal.log reveal-1.log
#cp devign.log devign-1.log
#cp all.log all-1.log
#sed -i.back -e 's/bash "train_reveal.sh"/bash "train_reveal_end.sh"/g' -e 's/bash "train_devign.sh"/#bash "train_devign.sh"/g' init.sh
