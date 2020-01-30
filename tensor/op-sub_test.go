package tensor

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_sub_forward(t *testing.T) {
	a := Constant([][]float64{{3, 4}})
	b := Constant([][]float64{{1, 2}})
	c := Sub(a, b)

	c.forward()

	assert.Equal(t, [][]float64{{2, 2}}, c.mat)
}

func Test_sub_backward(t *testing.T) {
	a := Constant([][]float64{{3, 4}, {5, 3}})
	a.isGradientEnabled = true
	b := Constant([][]float64{{1, 2}, {1, 4}})
	b.isGradientEnabled = true
	c := Sub(a, b)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	assert.Equal(t, [][]float64{{1, 1}, {1, 1}}, a.grad)
	assert.Equal(t, [][]float64{{-1, -1}, {-1, -1}}, b.grad)
}

func Test_sub_backward_isGradientsAreDisabled(t *testing.T) {
	a := Constant([][]float64{{3, 4}, {5, 3}})
	b := Constant([][]float64{{1, 2}, {1, 4}})
	c := Sub(a, b)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
