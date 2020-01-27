package tensor

import "github.com/gokadin/ml-framework/mat"

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
	for i := range od.a.mat {
		for j := 0; j < len(od.b.mat[0]); j++ {
			for k := range od.b.mat {
				tensor.mat[i][j] += od.a.mat[i][k] * od.b.mat[k][j]
			}
		}
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
