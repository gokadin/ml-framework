package data

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

func readCsv(filename string, startIndex, endIndex, limit int) [][]float64 {
	data := make([][]float64, 0)
    file := readFile(filename)
    defer file.Close()
    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
    	log.Fatal("Could not read CSV file", filename)
	}

    for i, row := range records {
    	if limit != -1 && i > limit {
    		break
		}

    	colIndex := 0
    	for j, value := range row {
    		if (startIndex > -1 && j < startIndex) || (endIndex != -1 && j > endIndex) {
    			continue
			}

            convertedValue, err := strconv.ParseFloat(value, 64)
            if err != nil {
            	log.Fatal("Could not parse one of the values in the CSV file", filename)
			}
            data[i][colIndex] = convertedValue
            colIndex++
		}
	}

	return data
}

func readFile(filename string) *os.File {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal("Could not open file", filename)
	}
	return file
}
