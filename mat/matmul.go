package mat

import "runtime"

func Matmul(a, b [][]float64) [][]float64 {
	start := make(chan int, len(a))
	output := make(chan matmulResult)
	for i := 0; i < runtime.NumCPU(); i++ {
		go thread(a, b, start, output)
	}

	resultSizeI := len(a)
	result := make([][]float64, resultSizeI)
	for i := range a {
		start <- i
	}

	count := 0
	for cr := range output {
		result[cr.i] = cr.row
		count++
		if count == len(a) {
			close(start)
			close(output)
		}
	}

	return result
}

func thread(a, b [][]float64, start chan int, output chan matmulResult) {
	for i := range start {
		row := make([]float64, len(b[0]))
		for j := 0; j < len(b[0]); j++ {
			for k := range b {
				row[j] += a[i][k] * b[k][j]
			}
		}
		output <- matmulResult{row, i}
	}
}

type matmulResult struct {
	row []float64
	i int
}
