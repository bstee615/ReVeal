#function devign()
#{
#  echo Model: Devign
#  paste <(ls -d $1/fold_* | rev | cut -d'/' -f1 | rev) <(ls $1/fold_*/models/devign-devign.log | sort | xargs grep -h 'Test Accuracy' | sed -e 's/Accuracy: //g' -e 's/Precision: //g' -e 's/Recall: //g' -e 's/F1: //g' -e 's/.*Test //g') | column -s $'\t' -t
#}
#
#function reveal()
#{
#  echo Model: Reveal
#  paste <(ls -d $1/fold_* | rev | cut -d'/' -f1 | rev) <(ls $1/fold_*/models/*.tsv | sort | xargs cat | sed -e 's/ /\t/g' -e 's/.*Test:\t//g') | column -s $'\t' -t
#}


function devign()
{
  echo Model:Devign
  echo Accuracy Precision Recall F1
  ls $1/fold_*/models/devign-devign.log | sort | xargs grep -h 'Test Accuracy' | sed -e 's/\t/ /g' -e 's/Accuracy: //g' -e 's/Precision: //g' -e 's/Recall: //g' -e 's/F1: //g' -e 's/.*Test //g'
}

function reveal()
{
  echo Model:Reveal
  echo Accuracy Precision Recall F1
  ls $1/fold_*/models/*.tsv | sort | xargs cat | sed -e 's/\t/ /g' -e 's/Test: //g'
}

function both()
{
  echo Dataset:$1
  devign $1
  reveal $1
}

[ $# -eq 0 ] && echo "Usage: $0 <directory_to_analyze>" && exit
[ ! -d $1 ] && echo "Does not exist: $1" && exit
both $1
