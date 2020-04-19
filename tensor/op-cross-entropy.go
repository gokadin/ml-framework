package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lcrossentropy
//#include <cross-entropy.h>
import "C"

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
	tensor.adjustShape(Shape{1, 1})
	C.cross_entropy(opw.a._tensor, opw.target._tensor, tensor._tensor)
}

func (opw *opCrossEntropy) backward(tensor *Tensor) {
	expandedGrad := mat.Expand(mat.Expand(mat.NewMat32f(mat.WithShape(tensor.shape.X, tensor.shape.Y), tensor.GradientToFloat32()), 1, opw.a.shape.Y), 0, opw.a.shape.X)
	opw.a.SetGradient(mat.Mul(expandedGrad, mat.Sub(mat.NewMat32f(mat.WithShape(opw.a.shape.X, opw.a.shape.Y), opw.a.ToFloat32()), mat.NewMat32f(mat.WithShape(opw.target.shape.X, opw.target.shape.Y), opw.target.ToFloat32()))).Data())
}

func CrossEntropy(pred, target *Tensor) *Tensor {
	result := Variable(1, 1)
	result.op = &opCrossEntropy{pred, target}
	return result
}
