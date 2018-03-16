#!/usr/bin/env bash

mkdir -p ../datasets/thoracic
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -o ../datasets/adult/adult.data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -o ../datasets/adult/adult.test

mkdir -p ../datasets/abalone
curl http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data \
    --output ../datasets/abalone/abalone.data

mkdir -p ../datasets/thoracic
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip \
    -o ../datasets/bank_marketing/bank-additional.zip
unzip ../datasets/bank_marketing/bank-additional.zip -d ../datasets/bank_marketing

sed -i '' 's/;/,/g' ../datasets/bank_marketing/bank-additional/bank-additional.csv
sed -i '' 's/;/,/g' ../datasets/bank_marketing/bank-additional/bank-additional-full.csv

mkdir -p ../datasets/credit_card
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls \
    -o ../datasets/credit_card/data.xls

mkdir -p ../datasets/thoracic
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff \
    --output ../datasets/thoracic/data.arff

mkdir -p ../datasets/wine_quality
curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv \
    --output ../datasets/wine_quality/winequality-red.csv
curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv \
    --output ../datasets/wine_quality/winequality-white.csv
curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names \
    --output ../datasets/wine_quality/winequality.names

root_path=`realpath ../`

export PYTHONPATH=${PYTHONPATH}:${root_path}

for process_file in "abalone" "adult" "bank_marketing" "credit_card" "thoracic" "wine_quality"
do
python process_${process_file}.py
done