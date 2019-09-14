package tensor

import (
	"math"
	"testing"
)

func Test_Sigmoid(t *testing.T) {
	a := NewTensor([][]float64{{1}})

	b := a.Sigmoid()

	expected := [][]float64{{1 / (1 + math.Pow(math.E, -1))}}
	if !equals(b.mat, expected) {
		t.Fatal("matrices are not equals for tensor b:", b.mat, "!=", expected)
	}
}
