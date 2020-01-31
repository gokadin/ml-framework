package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"runtime"
)

const operationDot = "opDot"

type opDot struct {
	a, b *Tensor
}

func (od *opDot) name() string {
	return operationDot
}

func (od *opDot) dependencies() []*Tensor {
	return []*Tensor{od.a, od.b}
}

func (od *opDot) forward(tensor *Tensor) {
	resultSizeI := len(od.a.mat)
	resultSizeJ := len(od.b.mat[0])
	result := make([][]float64, resultSizeI)

	start := make(chan int, len(od.a.mat))
	output := make(chan bool)
	for i := 0; i < runtime.NumCPU(); i++ {
		go thread(od.a.mat, od.b.mat, start, output, result)
	}

	for i := range od.a.mat {
		result[i] = make([]float64, resultSizeJ)
		start <- i
	}

	count := 0
	for range output {
		count++
		if count == len(od.a.mat) {
			close(start)
			close(output)
			tensor.mat = result
		}
	}
}

func thread(a, b [][]float64, start chan int, output chan bool, mat [][]float64) {
	for i := range start {
		for j := 0; j < len(b[0]); j++ {
			for k := range b {
				mat[i][j] += a[i][k] * b[k][j]
			}
		}
		output <- true
	}
}

func (od *opDot) backward(tensor *Tensor) {
	if od.a.isGradientEnabled {
		od.a.grad = mat.Dot(tensor.grad, mat.Transpose(od.b.mat))
	}
	if od.b.isGradientEnabled {
		od.b.grad = mat.Transpose(mat.Dot(mat.Transpose(tensor.grad), od.a.mat))
	}
}

func Dot(a, b *Tensor) *Tensor {
	result := Variable(len(a.mat), len(b.mat[0]))
	result.op = &opDot{a, b}
	return result
}
