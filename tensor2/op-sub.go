package tensor2

const operationSub = "opSub"

type opSub struct {
	a, b *Tensor
}

func (os *opSub) name() string {
	return operationSub
}

func (os *opSub) dependencies() []*Tensor {
	return []*Tensor{os.a, os.b}
}

func (os *opSub) forward(mat [][]float64) {
	for i := range mat {
		for j := range mat[i] {
			mat[i][j] = os.a.mat[i][j] - os.b.mat[i][j]
		}
	}
}

func (os *opSub) backward() {

}

func Sub(a, b *Tensor) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opSub{a, b}
	return result
}
