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

func (a *Autograd) pruneGraph(c, f *Tensor) *operation {
    for _, children := range f.operation.children {
        if children.tensor.id == c.id {

		}
	}
}