package tensor2

const operationAdd = "opAdd"

type opAdd struct {
	a, b *Tensor
}

func (oa *opAdd) name() string {
	return operationAdd
}

func (oa *opAdd) dependencies() []*Tensor {
	return []*Tensor{oa.a, oa.b}
}

func (oa *opAdd) forward(mat [][]float64) {
	for i := range mat {
		for j := range mat[i] {
			mat[i][j] = oa.a.mat[i][j] + oa.b.mat[i][j]
		}
	}
}

func (oa *opAdd) backward(grad [][]float64) {
	if oa.a.isGradientEnabled {
		oa.a.grad = grad
	}

	if oa.b.isGradientEnabled {
		oa.b.grad = grad
	}
}

func Add(a, b *Tensor) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opAdd{a, b}
	return result
}
