package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_Expand_forward(t *testing.T) {
	a := Variable(mat.WithShape(1, 3)).SetData([]float32{1, 2, 3})
	c := Expand(a, 0, 3)

	c.forward()

	assert.Equal(t, []float32{1, 2, 3, 1, 2, 3, 1, 2, 3}, c.TempData())
}

func Test_Expand_forwardLargeNumbers(t *testing.T) {
	a := Variable(mat.WithShape(1, 3)).SetData([]float32{1, 2, 3})
	c := Expand(a, 0, 3000)

	c.forward()

	expected := make([]float32, 3 * 3000)
	for i := 0; i < 3 * 3000; i += 3 {
		expected[i] = 1
		expected[i + 1] = 2
		expected[i + 2] = 3
	}
	assert.Equal(t, expected, c.TempData())
}

func Test_Expand_forwardLargeNumbersEvenBigger(t *testing.T) {
	aMat := make([]float32, 6000)
	for i := 0; i < 6000; i++ {
		aMat[i] = float32(i)
	}
	a := Variable(mat.WithShape(1, 6000)).SetData(aMat)
	c := Expand(a, 0, 3000)

	c.forward()

	expected := make([]float32, 3000 * 6000)
	for i := 0; i < 3000; i++ {
		for j := 0; j < 6000; j++ {
			expected[i * 6000 + j] = aMat[j]
		}
	}
	assert.Equal(t, expected, c.TempData())
}
