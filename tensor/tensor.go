package tensor

import (
	"github.com/google/uuid"
	"math/rand"
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

func NewRandomTensor(x, y int) *Tensor {
    mat := make([][]float64, x)
    for i := range mat {
    	mat[i] = make([]float64, y)
    	for j := range mat[i] {
    		mat[i][j] = rand.Float64()
		}
	}

    return NewTensor(mat)
}

func (t *Tensor) Data() [][]float64 {
	return t.mat
}

func (t *Tensor) Reduce(coefficient float64) {
	t.mat = sub(t.mat, mulScalar(t.operation.gradient, coefficient))
}

func (t *Tensor) Equals(other *Tensor) bool {
	return equals(t.mat, other.mat)
}

func (t *Tensor) DisableGradient() {
	t.isGradientEnabled = false
}
