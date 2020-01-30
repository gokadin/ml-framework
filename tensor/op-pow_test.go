package tensor

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_pow_forward(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	c := Pow(a, 3)

	c.forward()

	assert.Equal(t, [][]float64{{1, 8}, {27, 64}}, c.mat)
}

func Test_pow_backward(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	a.isGradientEnabled = true
	c := Pow(a, 3)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	assert.Equal(t, [][]float64{{3, 12}, {27, 48}}, a.grad)
}

func Test_pow_backward_forPowerOfTwo(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	a.isGradientEnabled = true
	c := Pow(a, 2)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	assert.Equal(t, [][]float64{{2, 4}, {6, 8}}, a.grad)
}
