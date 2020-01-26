package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_Autograd_operationName(t *testing.T) {
	a := NewTensor([][]float64{{1, 1}, {1, 1}})
	b := NewTensor([][]float64{{1, 1}, {1, 1}})

	c := Add(a, b)

	assert.Equal(t, operationAdd, c.operation.name)
}

func Test_Autograd_operationChildren(t *testing.T) {
	a := NewTensor([][]float64{{1, 1}, {1, 1}})
	b := NewTensor([][]float64{{1, 1}, {1, 1}})

	c := Add(a, b)

	assert.Equal(t, 2, len(c.operation.children))
}

func Test_Autograd_operationChildrenAreLeaf(t *testing.T) {
	a := NewTensor([][]float64{{1, 1}, {1, 1}})
	b := NewTensor([][]float64{{1, 1}, {1, 1}})

	c := Add(a, b)

	assert.True(t, c.operation.children[0].isLeaf())
	assert.True(t, c.operation.children[1].isLeaf())
}

func Test_Tensor_Softmax(t *testing.T) {
	a := NewTensor([][]float64{{2, 48, 50}})

	s := a.Softmax()

	assert.Equal(t, [][]float64{{1.0}}, mat.Sum(s.mat, 1))
}