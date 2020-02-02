package tensor

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_add_forward(t *testing.T) {
	a := Constant([][]float64{{1, 2}})
	b := Constant([][]float64{{2, 3}})
	c := Add(a, b)

	c.forward()

	assert.Equal(t, [][]float64{{3, 5}}, c.mat)
}

func Test_add_forward_multipleAssociations(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {2, 3}})
	b := Constant([][]float64{{2, 3}, {1, 2}})
	c := Add(a, b)

	c.forward()

	assert.Equal(t, [][]float64{{3, 5}, {3, 5}}, c.mat)
}

func Test_add_backward(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {2, 3}})
	a.isGradientEnabled = true
	b := Constant([][]float64{{2, 3}, {1, 2}})
	b.isGradientEnabled = true
	c := Add(a, b)
	c.forward()
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))

	c.backward()

	assert.Equal(t, [][]float64{{1, 1}, {1, 1}}, a.grad)
	assert.Equal(t, [][]float64{{1, 1}, {1, 1}}, b.grad)
}

func Test_add_backward_isGradientsAreDisabled(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {2, 3}})
	b := Constant([][]float64{{2, 3}, {1, 2}})
	c := Add(a, b)
	c.forward()
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
