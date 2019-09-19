package tensor

type Autograd struct {

}

func NewAutograd() *Autograd {
	return &Autograd{

	}
}

func (a *Autograd) Gradient(c, f *Tensor) [][]float64 {
	return [][]float64{{1}}
}

func isolate(c, f *Tensor) *operation {
	return findComponentOperation(c.operation, f.operation)
}

func findComponentOperation(leaf, root *operation) *operation {
	for _, child := range root.children {
		if child.tensor.id == leaf.tensor.id {
            op := newOperation(operationDifferentiateNone, child.tensor, []*operation{})
		}
	}
}