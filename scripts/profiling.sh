TEXT="bible.txt"
INPUT_FOLDER="./data/englishTexts"

GINFOLOG_FOLDER="./data/reports/log/ginfo"
PERF_FOLDER="./data/reports/log/perf"
OUTPUT_FOLDER="./data/reports/output"

mkdir -p $GINFOLOG_FOLDER
mkdir -p $OUTPUT_FOLDER

# Test on eight patterns
for i in {1..3};
do
    fileName=$i"-pattern.txt"

    # For each pattern, every algorithm is considered
    for j in {1..6};
    do
        # Four configurations are applied of g = {100, 200, 500, 1000}
        for m in 100 200 500 1000;
        do
            outFileName=$i"-pattern-"$j"-algo-"$m"-g_output.txt"
            logFileName=$i"-pattern-"$j"-algo-"$m"-g.log"

            if [ $j -eq 2 ] || [ $j -eq 4 ] || [ $j -eq 6 ]; then
                nvprof --log-file $GINFOLOG_FOLDER/$logFileName ./a.out $INPUT_FOLDER/$TEXT $INPUT_FOLDER/$fileName $j $m 8 > $OUTPUT_FOLDER/$outfileName
                nvprof -–metrics all $PERF_FOLDER/$logFileName ./a.out $INPUT_FOLDER/$TEXT $INPUT_FOLDER/$fileName $j $m 8
            else
                nvprof --log-file $GINFOLOG_FOLDER/$logFileName ./a.out $INPUT_FOLDER/$TEXT $INPUT_FOLDER/$fileName $j $m 1 > $OUTPUT_FOLDER/$outfileName
                nvprof -–metrics all $PERF_FOLDER/$logFileName ./a.out $INPUT_FOLDER/$TEXT $INPUT_FOLDER/$fileName $j $m 1
            fi
        done
    done
done