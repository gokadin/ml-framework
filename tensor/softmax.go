package tensor

//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -lm -lsoftmax
//#include <softmax.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationSoftmax = "opSoftmax"

type opSoftmax struct {
	a *Tensor
}

func (o *opSoftmax) name() string {
	return operationSoftmax
}

func (o *opSoftmax) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opSoftmax) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opSoftmax) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opSoftmax) forward(tensor *Tensor) {
	tensor.SetData(mat.Softmax(o.a.ToMat32f()).Data())
}

func (o *opSoftmax) backward(tensor *Tensor) {
	o.a.SetGradient(tensor.GradientToFloat32())
}

func Softmax(a *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opSoftmax{a}
	result._tensor.op = C.alloc_softmax(a._tensor)
	return result
}
