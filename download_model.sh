# get the directory where the script runs
# https://stackoverflow.com/questions/59895/get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done

DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
URL="http://people.csail.mit.edu/tianxiao/language-style-transfer/model"

mkdir -p $DIR/model

files=(
    "yelp.d100.emb.txt"
    "yelp.vocab"
    "model.data-00000-of-00001"
    "model.index"
    "model.meta"
)
echo $DIR

for file in "${files[@]}"; do
    wget $URL/$file -O $DIR/model/$file
done
