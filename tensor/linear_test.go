package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type linearTestCases struct {
	name string
	a *Tensor
	x *Tensor
	b *Tensor
	runOnGpu bool
}

func buildLinearTestCases() []linearTestCases {
	return []linearTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), OfShape(1, 1).SetData([]float32{3}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), OfShape(1, 1).SetData([]float32{3}), false},
		{"2x2 GPU", Ones(2, 2), Ones(2, 2), Ones(1, 2), true},
		{"2x2 CPU", Ones(2, 2), Ones(2, 2), Ones(1, 2), false},
	}
}

func Test_linear_forward(t *testing.T) {
	t.Parallel()
	for _, test := range buildLinearTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			expected := mat.Add(mat.MatMulParallel(test.a.ToMat32f(), test.x.ToMat32f()), mat.Expand(test.b.ToMat32f(), 0, test.a.Shape().X)).Data()
			c := Linear(test.a, test.x, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}
