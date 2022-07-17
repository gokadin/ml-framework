package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type softmaxTestCases struct {
	name     string
	a        *Tensor
	runOnGpu bool
}

func buildSoftmaxTestCases() []softmaxTestCases {
	return []softmaxTestCases{
		{"1x1 GPU", From(mat.InitRandom, 1, 1), true},
		{"1x1 CPU", From(mat.InitRandom, 1, 1), false},
		{"2x2 GPU", From(mat.InitRandom, 2, 2), true},
		{"2x2 CPU", From(mat.InitRandom, 2, 2), false},
		{"3x4 GPU", From(mat.InitRandom, 2, 2), true},
		{"3x4 CPU", From(mat.InitRandom, 2, 2), false},
		{"6x3 GPU", From(mat.InitRandom, 2, 2), true},
		{"6x3 CPU", From(mat.InitRandom, 2, 2), false},
		{"1200x2 ones GPU", From(mat.InitRandom, 1200, 2), true},
		{"1200x2 ones CPU", From(mat.InitRandom, 1200, 2), false},
		{"2x1200 ones GPU", From(mat.InitRandom, 2, 1200), true},
		{"2x1200 ones CPU", From(mat.InitRandom, 2, 1200), false},
		{"10x10 random GPU", From(mat.InitRandom, 10, 10), true},
		{"10x10 random CPU", From(mat.InitRandom, 10, 10), false},
	}
}

func Test_softmax_forward(t *testing.T) {
	for _, test := range buildSoftmaxTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Softmax(test.a.ToMat32f()).Data()
			c := Softmax(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.000001)
		})
	}
}
