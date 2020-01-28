package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"
)

func perftest() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	a := make([][]float64, 60000)
	for i := range a {
		arow := make([]float64, 784)
		for j := range arow {
			arow[j] = rand.NormFloat64()
		}
		a[i] = arow
	}

	b := make([][]float64, 784)
	for i := range b {
		brow := make([]float64, 128)
		for j := range brow {
			brow[j] = rand.NormFloat64()
		}
		b[i] = brow
	}

	t := time.Now().UnixNano()

	res := matmulConcurrent(a, b)
	_ = res

	diff := time.Now().UnixNano() - t
	fmt.Println("Time:", diff / int64(time.Millisecond))
}

func matmul(a, b [][]float64) [][]float64 {
	resultSizeI := len(a)
	resultSizeJ := len(b[0])
	result := make([][]float64, resultSizeI)
	for i := range a {
		result[i] = make([]float64, resultSizeJ)
		for j := 0; j < len(b[0]); j++ {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

type conret struct {
	row []float64
	i int
}

func matmulConcurrent(a, b [][]float64) [][]float64 {
	start := make(chan int, 60000)
	output := make(chan conret)
	for i := 0; i < 24; i++ {
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

func thread(a, b [][]float64, start chan int, output chan conret) {
	for i := range start {
		row := make([]float64, len(b[0]))
		for j := 0; j < len(b[0]); j++ {
			for k := range b {
				row[j] += a[i][k] * b[k][j]
			}
		}
		output <- conret{row, i}
	}
}
