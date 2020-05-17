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

	sums := make([]float32, tensor.GradShape().Size())
	for i := 0; i < tensor.GradShape().X; i++ {
		var partialSum float32
		for j := 0; j < tensor.GradShape().Y; j++ {
			partialSum += tensorGrad.Data()[i * tensor.GradShape().Y + j]
		}
		for j := 0; j < tensor.GradShape().Y; j++ {
			sums[i * tensor.GradShape().Y + j] = partialSum
		}
	}

	o.a.SetGradient(mat.Sub(tensorGrad, mat.Mul(mat.Exp(tensorMat), mat.NewMat32f(mat.WithShape(tensor.GradShape().X, tensor.GradShape().Y), sums))).Data())
}

func LogSoftmax(a *Tensor) *Tensor {
	o := &opLogSoftmax{a}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
