package tensor

import "github.com/gokadin/ml-framework/mat"

const operationSub = "opSub"

type opSub struct {
	a, b *Tensor
}

func (os *opSub) name() string {
	return operationSub
}

func (os *opSub) dependencies() []*Tensor {
	return []*Tensor{os.a, os.b}
}

func (os *opSub) forward(tensor *Tensor) {
	tensor.SetData(mat.Sub(os.a.ToMat32f(), os.b.ToMat32f()).Data())
}

func (os *opSub) backward(tensor *Tensor) {
	if os.a.isGradientEnabled {
		os.a.SetGradient(tensor.GradientToFloat32())
	}

	if os.b.isGradientEnabled {
		os.b.SetGradient(mat.Neg(tensor.GradientToMat32()).Data())
	}
}

func Sub(a, b *Tensor) *Tensor {
	result := Variable(a.Shape().X, a.Shape().Y)
	result.op = &opSub{a, b}
	return result
}
