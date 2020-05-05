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

func (opw *opSoftmax) name() string {
	return operationSoftmax
}

func (opw *opSoftmax) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opSoftmax) forward(tensor *Tensor) {
	tensor.SetData(mat.Softmax(opw.a.ToMat32f()).Data())
}

func (opw *opSoftmax) backward(tensor *Tensor) {
	opw.a.SetGradient(tensor.GradientToFloat32())
}

func Softmax(a *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opSoftmax{a}
	result._tensor.op = C.alloc_softmax(a._tensor)
	return result
}
