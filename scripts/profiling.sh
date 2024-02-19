INPUT_FOLDER="./data/englishTexts"
TEXT="bible.txt"
LOG_FOLDER="./data/reports/log"
OUTPUT_FOLDER="./data/reports/output"
mkdir -p $LOG_FOLDER
mkdir -p $OUTPUT_FOLDER

# Test on eight patterns
for i in {1..2};
do
    fileName=$i"-pattern.txt"

    # For each pattern, every algorithm is considered
    for j in {1..6};
    do
        # Three configurations are applied of g = {200, 500, 1000}
        for m in 200 500 1000;
        do
            outFileName=$i"-pattern-"$j"-algo-"$m"-g_output.txt"
            logFileName=$i"-pattern-"$j"-algo-"$m"-g_gpu-trace.txt"

            if [ $j -eq 2 ] || [ $j -eq 4 ] || [ $j -eq 6 ]; then
                nvprof --print-gpu-trace --log-file $LOG_FOLDER/$logFileName ./a.out $INPUT_FOLDER/$TEXT $INPUT_FOLDER/$fileName $j $m 8 > $OUTPUT_FOLDER/$outfileName
            else
                nvprof --print-gpu-trace --log-file $LOG_FOLDER/$logFileName ./a.out $INPUT_FOLDER/$TEXT $INPUT_FOLDER/$fileName $j $m 1 > $OUTPUT_FOLDER/$outfileName
            fi
        done
    done
done