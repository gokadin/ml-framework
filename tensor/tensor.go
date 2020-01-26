package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/google/uuid"
)

type Tensor struct {
	id string
	mat [][]float64
	isGradientEnabled bool
	operation *operation
}

func NewTensor(mat [][]float64) *Tensor {
	t := &Tensor{
		id: uuid.New().String(),
        mat: mat,
        isGradientEnabled: true,
	}
	t.operation = newOperation(operationNone, t, []*operation{})
	return t
}

func (t *Tensor) Id() string {
	return t.id
}

func (t *Tensor) Data() [][]float64 {
	return t.mat
}

func (t *Tensor) Reduce(grad [][]float64) {
	t.mat = mat.Sub(t.mat, grad)
}

func (t *Tensor) Gradient() [][]float64 {
    return t.operation.gradient
}

func (t *Tensor) DisableGradient() {
	t.isGradientEnabled = false
}

func (t *Tensor) Backward() {
	t.operation.gradient = generateIdentityGradient(t.mat)
	t.backpropagate()
}

func (t *Tensor) backpropagate() {
	t.operation.differentiate(t.operation.gradient)
	for _, child := range t.operation.children {
		if !child.isLeaf() {
			child.tensor.backpropagate()
		}
	}
}
