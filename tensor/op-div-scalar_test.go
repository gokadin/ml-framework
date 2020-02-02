package tensor

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_divScalar_forward(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	c := DivScalar(a, 2)

	c.forward()

	assert.Equal(t, [][]float64{{0.5, 1}, {1.5, 2}}, c.mat)
}

func Test_divScalar_backward(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	a.isGradientEnabled = true
	c := DivScalar(a, 2)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	assert.Equal(t, [][]float64{{0.5, 0.5}, {0.5, 0.5}}, a.grad)
}
