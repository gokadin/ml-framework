package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationCrossEntropy = "opCrossEntropy"

type opCrossEntropy struct {
	a *Tensor
	target *Tensor
}

func (opw *opCrossEntropy) name() string {
	return operationCrossEntropy
}

func (opw *opCrossEntropy) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opCrossEntropy) forward(tensor *Tensor) {
	tensor.mat = mat.DivScalar(mat.Sum(mat.Neg(mat.Log(mat.Sum(mat.Mul(opw.target.mat, opw.a.mat), 1))), 0), float32(opw.a.mat.Shape().X))
}

func (opw *opCrossEntropy) backward(tensor *Tensor) {
	expandedGrad := mat.Expand(mat.Expand(tensor.grad, 1, opw.a.mat.Shape().Y), 0, opw.a.mat.Shape().X)
	opw.a.grad = mat.Mul(expandedGrad, mat.Sub(opw.a.mat, opw.target.mat))
}

func CrossEntropy(pred, target *Tensor) *Tensor {
	result := Variable(mat.WithShape(1, 1))
	result.op = &opCrossEntropy{pred, target}
	return result
}
