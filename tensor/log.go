package tensor

import (
	"ml-framework/mat"
)

const operationLog = "opLog"

type opLog struct {
	a *Tensor
}

func (o *opLog) name() string {
	return operationLog
}

func (o *opLog) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opLog) forwardShape() mat.Shape {
	return o.a.Shape()
}

func (o *opLog) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{tensorShape}
}

func (o *opLog) forward(tensor *Tensor) {
	tensor.SetData(mat.Log(o.a.ToMat32f()).Data())
}

func (o *opLog) backward(tensor *Tensor) {
	o.a.SetGradient(mat.DivScalarBy(tensor.GradientToMat32(), 1).Data())
}

func Log(a *Tensor) *Tensor {
	o := &opLog{a}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
