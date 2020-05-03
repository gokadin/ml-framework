package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type softmaxTestCases struct {
	name string
	a *Tensor
	runOnGpu bool
}

func buildSoftmaxTestCases() []softmaxTestCases {
	return []softmaxTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), false},
		{"1200x2 ones GPU", Ones(1200, 2), true},
		{"1200x2 ones CPU", Ones(1200, 2), false},
		{"2x1200 ones GPU", Ones(2, 1200), true},
		{"2x1200 ones CPU", Ones(2, 1200), false},
		{"10x10 random GPU", From(InitRandom, 10, 10), true},
		{"10x10 random GPU", From(InitRandom, 10, 10), false},
	}
}

func Test_softmax_forward(t *testing.T) {
	t.Parallel()
	for _, test := range buildSoftmaxTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			expected := mat.Softmax(test.a.ToMat32f()).Data()
			c := Softmax(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.00001)
		})
	}
}
