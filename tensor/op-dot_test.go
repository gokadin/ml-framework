package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_dot_forward_simple(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	b := Constant([][]float64{{5, 6}, {7, 8}})
	c := Dot(a, b)

	c.forward()

	assert.Equal(t, [][]float64{{19, 22}, {43, 50}}, c.mat)
}

func Test_dot_forward_differentSimple(t *testing.T) {
	a := Constant([][]float64{{1, 0}, {0, 1}})
	b := Constant([][]float64{{4, 1}, {2, 2}})
	c := Dot(a, b)

	c.forward()

	assert.Equal(t, [][]float64{{4, 1}, {2, 2}}, c.mat)
}

func Test_dot_backward(t *testing.T) {
	a := Constant([][]float64{{1, 0}, {0, 1}})
	a.isGradientEnabled = true
	b := Constant([][]float64{{4, 1}, {2, 2}})
	b.isGradientEnabled = true
	c := Dot(a, b)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	assert.Equal(t, mat.Dot(c.grad, mat.Transpose(b.mat)), a.grad)
	assert.Equal(t, mat.Transpose(mat.Dot(mat.Transpose(c.grad), a.mat)), b.grad)
}

func Test_dot_backward_isGradientsAreDisabled(t *testing.T) {
	a := Constant([][]float64{{1, 0}, {0, 1}})
	b := Constant([][]float64{{4, 1}, {2, 2}})
	c := Dot(a, b)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
