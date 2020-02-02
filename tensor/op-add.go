package tensor

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

func (oa *opAdd) forward(tensor *Tensor) {
	for i := range tensor.mat {
		for j := range tensor.mat[i] {
			tensor.mat[i][j] = oa.a.mat[i][j] + oa.b.mat[i][j]
		}
	}
}


func (oa *opAdd) backward(tensor *Tensor) {
	if oa.a.isGradientEnabled {
		oa.a.grad = tensor.grad
	}

	if oa.b.isGradientEnabled {
		oa.b.grad = tensor.grad
	}
}

func Add(a, b *Tensor) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opAdd{a, b}
	return result
}
