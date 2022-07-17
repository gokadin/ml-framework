package tensor

//#include <binary-cross-entropy.h>
import "C"
import "ml-framework/mat"

const operationBinaryCrossEntropy = "opBinaryCrossEntropy"

type opBinaryCrossEntropy struct {
	a      *Tensor
	target *Tensor
}

func (o *opBinaryCrossEntropy) name() string {
	return operationBinaryCrossEntropy
}

func (o *opBinaryCrossEntropy) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opBinaryCrossEntropy) forwardShape() mat.Shape {
	return mat.Dim(1, 1)
}

func (o *opBinaryCrossEntropy) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{o.a.Shape()}
}

func (o *opBinaryCrossEntropy) forward(tensor *Tensor) {
	if !o.a.Shape().Equals(o.target.Shape()) {
		handleIncompatibleShapes("binary-cross-entropy", o.a.Shape(), o.target.Shape())
	}
	C.bce_forward(tensor._tensor, o.a._tensor, o.target._tensor)
}

func (o *opBinaryCrossEntropy) backward(tensor *Tensor) {
	C.bce_backward(tensor._tensor, o.a._tensor, o.target._tensor)
}

func BinaryCrossEntropy(pred, target *Tensor) *Tensor {
	o := &opBinaryCrossEntropy{pred, target}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
