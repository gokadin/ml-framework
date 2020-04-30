package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lrelu
//#include <relu.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationRelu = "opRelu"

type opRelu struct {
	a *Tensor
}

func (opw *opRelu) name() string {
	return operationRelu
}

func (opw *opRelu) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opRelu) forward(tensor *Tensor) {
	C.relu(opw.a._tensor, tensor._tensor)
	//tensor.SetData(mat.Apply(tensor.ToMat32f(), func(value float32) float32 {
	//	if value > 0 {
	//		return value
	//	}
	//	return 0
	//}).Data())
}

func (opw *opRelu) backward(tensor *Tensor) {
	d := mat.Apply(tensor.ToMat32f(), func(value float32) float32 {
		if value > 0 {
			return 1
		}
		return 0
	})
	opw.a.SetGradient(mat.Mul(tensor.GradientToMat32(), d).Data())
}

func Relu(a *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opRelu{a}
	return result
}
