package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationLogSoftmax = "opLogSoftmax"

type opLogSoftmax struct {
	a *Tensor
}

func (o *opLogSoftmax) name() string {
	return operationLogSoftmax
}

func (o *opLogSoftmax) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opLogSoftmax) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opLogSoftmax) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opLogSoftmax) forward(tensor *Tensor) {
	tensor.SetData(mat.Log(mat.Softmax(o.a.ToMat32f())).Data())
}

func (o *opLogSoftmax) backward(tensor *Tensor) {
	tensorMat := tensor.ToMat32f()
	tensorGrad := tensor.GradientToMat32()

	sums := make([]float32, tensor.Shape().Size())
	for i := 0; i < tensor.Shape().X; i++ {
		var partialSum float32
		for j := 0; j < tensor.Shape().Y; j++ {
			partialSum += tensorGrad.Data()[i * tensor.Shape().Y + j]
		}
		for j := 0; j < tensor.Shape().Y; j++ {
			sums[i * tensor.Shape().Y + j] = partialSum
		}
	}

	o.a.SetGradient(mat.Sub(tensorGrad, mat.Mul(mat.Exp(tensorMat), mat.NewMat32f(mat.WithShape(tensor.Shape().X, tensor.Shape().Y), sums))).Data())
}

func LogSoftmax(a *Tensor) *Tensor {
	o := &opLogSoftmax{a}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
