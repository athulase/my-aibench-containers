#!/bin/bash

source ~/.bashrc

LOG_DIR=$AMUNMT_LOG_DIR
minibatch=$MARIAN_MINIBATCH
maxibatch=$MARIAN_MAXIBATCH
nr_threads=$MARIAN_THREADS

export OMP_NUM_THREADS=$MARIAN_OMP_THREADS
export MKL_NUM_THREADS=$MARIAN_MKL_THREADS

mkdir -p ${LOG_DIR}

cd /marian/examples/translate

cat data/newstest2015.ende.en |     moses-scripts/scripts/tokenizer/normalize-punctuation.perl -l en > ${LOG_DIR}/normalized_en.txt 2> ${LOG_DIR}/normalized_en.err
cat ${LOG_DIR}/normalized_en.txt | moses-scripts/scripts/tokenizer/tokenizer.perl -l en -penn > ${LOG_DIR}/tokenized_en.txt 2> ${LOG_DIR}/tokenized_en.err
cat ${LOG_DIR}/tokenized_en.txt | moses-scripts/scripts/recaser/truecase.perl -model en-de/truecase-model.en > ${LOG_DIR}/true_case_en_de.txt 2> ${LOG_DIR}/true_case_en_de.err
cat ${LOG_DIR}/true_case_en_de.txt |  ../../build/amun -m en-de/model.npz -s en-de/vocab.en.json -t en-de/vocab.de.json     --mini-batch ${minibatch} --maxi-batch ${maxibatch} -b 12 -n --bpe en-de/ende.bpe --cpu-threads ${nr_threads} > ${LOG_DIR}/amun_translate.txt 2> ${LOG_DIR}/amun_translate.err
cat ${LOG_DIR}/amun_translate.txt | moses-scripts/scripts/recaser/detruecase.perl > ${LOG_DIR}/translated_detruecase.txt 2>  ${LOG_DIR}/translated_detruecase.err
cat ${LOG_DIR}/translated_detruecase.txt | moses-scripts/scripts/tokenizer/detokenizer.perl -l de >   ${LOG_DIR}/newstest2015.out 2>   ${LOG_DIR}/newstest2015.err

timetaken=`tail -2 ${LOG_DIR}/amun_translate.err | grep wall | awk -F wall '{print $1}' | awk -F: '{print $4}' | sed 's/s//g' | sed 's/ //g'`
echo $timetaken

total_words=$(wc -w ${LOG_DIR}/true_case_en_de.txt | awk '{print $1}')
result=$(echo ${total_words}/${timetaken}|bc -l)

ts=$(date +%s)
echo "WORDS/SEC=${result}" > ${LOG_DIR}/amunmt_wps_${HOSTNAME}_${ts}_${nr_threads}_${minibatch}.txt

# Save the throughput
echo "$result WordsPerSecond" >> /relevant_metrics
echo "1.0 AverageConfidence" >> /relevant_metrics
cp /relevant_metrics $LOG_DIR/relevant_metrics