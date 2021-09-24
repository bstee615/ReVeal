if [ $# -lt 2 ] || [ -z "$1" ] || [ -z "$2" ]
then
  echo "Usage: $0 <model> <directory_to_analyze>" && exit
fi

train_script="train_$1.sh"
if [ ! -f "$train_script" ]
then
  echo "Does not exist: $train_script" && exit
fi

if [ ! -d "$2" ]
then
  echo "Does not exist: $2" && exit
fi

name="$1-$(echo "$2" | rev | cut -d'/' -f1 | rev)"
echo "Name: $name"
sbatch -J "$name" --output "sbatch-%j-$name.out" batch.sh bash "$train_script" "$2"
