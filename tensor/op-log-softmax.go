package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationLogSoftmax = "opLogSoftmax"

type opLogSoftmax struct {
	a *Tensor
}

func (opw *opLogSoftmax) name() string {
	return operationLogSoftmax
}

func (opw *opLogSoftmax) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opLogSoftmax) forward(tensor *Tensor) {
	tensor.adjustShape(opw.a.shape)
	tensor.SetData(mat.Log(mat.Softmax(opw.a.ToMat32f())).Data())
}

func (opw *opLogSoftmax) backward(tensor *Tensor) {
	tensorMat := tensor.ToMat32f()
	tensorGrad := tensor.GradientToMat32()

	sums := make([]float32, tensor.shape.Size())
	for i := 0; i < tensor.shape.X; i++ {
		var partialSum float32
		for j := 0; j < tensor.shape.Y; j++ {
			partialSum += tensorGrad.Data()[i * tensor.shape.Y + j]
		}
		for j := 0; j < tensor.shape.Y; j++ {
			sums[i * tensor.shape.Y + j] = partialSum
		}
	}

	opw.a.SetGradient(mat.Sub(tensorGrad, mat.Mul(mat.Exp(tensorMat), mat.NewMat32f(mat.WithShape(tensor.shape.X, tensor.shape.Y), sums))).Data())
}

func LogSoftmax(a *Tensor) *Tensor {
	result := Variable(a.shape.X, a.shape.Y)
	result.op = &opLogSoftmax{a}
	return result
}
