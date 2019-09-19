package tensor

import (
	"github.com/google/uuid"
	"math/rand"
)

type Tensor struct {
	id string
	mat [][]float64
	grad [][]float64
	operations []*operation
}

func NewTensor(mat [][]float64) *Tensor {
	return &Tensor{
		id: uuid.New().String(),
        mat: mat,
        operations: make([]*operation, 0),
	}
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

func (t *Tensor) Equals(other *Tensor) bool {
	return equals(t.mat, other.mat)
}

func (t *Tensor) ResetGradient() {
	t.grad = nil
}

func (t *Tensor) Reduce(coefficient float64) {
	t.mat = sub(t.mat, mulScalar(t.grad, coefficient))
}