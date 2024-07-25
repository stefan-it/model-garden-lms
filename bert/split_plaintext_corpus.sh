# The final, splitted pretraining corpus output folder
PRETRAINING_CORPUS_PATH=./pretraining-corpus

# Path, where the extracted 000{00..14}.txt files are located at
EXTRACTED_CORPUS_PATH=./extracted-plaintext-corpus

# We use 400M chunks (it is maybe a hyper-parameter or not...)
SPLIT_SIZE=400M

mkdir -p ${PRETRAINING_CORPUS_PATH}

for index in $(seq -w 0 14)
do
    echo Splitting 000${index}.txt ...

    mkdir -p ${PRETRAINING_CORPUS_PATH}/part-${index}

    split -C ${SPLIT_SIZE} -d ${EXTRACTED_CORPUS_PATH}/000${index}.txt ${PRETRAINING_CORPUS_PATH}/part-${index}/part-${index}-
done
