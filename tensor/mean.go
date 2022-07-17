package tensor

import "ml-framework/mat"

const operationMean = "opMean"

type opMean struct {
	a *Tensor
}

func (o *opMean) name() string {
	return operationMean
}

func (o *opMean) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opMean) forwardShape() mat.Shape {
	return o.a.Shape()
}

func (o *opMean) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{tensorShape, tensorShape}
}

func (o *opMean) forward(tensor *Tensor) {
	return
}

func (o *opMean) backward(tensor *Tensor) {
	return
}

func Mean(a *Tensor) *Tensor {
	o := &opMean{a}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
