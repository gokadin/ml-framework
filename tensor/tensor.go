package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/google/uuid"
)

type Tensor struct {
	id string
	name string
	op op
	mat *mat.Mat32f
	grad *mat.Mat32f
	isGradientEnabled bool
}

func Constant(mat *mat.Mat32f) *Tensor {
	return &Tensor{
		id: uuid.New().String(),
		mat: mat,
	}
}

func Variable(shape mat.Shape) *Tensor {
	return &Tensor {
		id: uuid.New().String(),
		mat: mat.NewMat32f(shape, nil),
	}
}

func (t *Tensor) Id() string {
	return t.id
}

func (t *Tensor) SetData(data *mat.Mat32f) {
	t.mat = data
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) SetName(name string) *Tensor {
	t.name = name
	return t
}

func (t *Tensor) Data() *mat.Mat32f {
	return t.mat
}

func (t *Tensor) Gradient() *mat.Mat32f {
	return t.grad
}

func (t *Tensor) Reduce(grad *mat.Mat32f) {
	t.mat.Sub(grad)
}

func (t *Tensor) Shape() mat.Shape {
	return t.mat.Shape()
}

func (t *Tensor) Reshape(shape mat.Shape) {
	t.mat.Reshape(shape)
}

func (t *Tensor) forward() {
	t.op.forward(t)
}

func (t *Tensor) backward() {
	t.op.backward(t)
}
