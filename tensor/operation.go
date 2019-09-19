package tensor

const (
	operationNone = "operationNone"
	operationAdd = "operationAdd"
	operationSub = "operationSub"

	operationDifferentiateNone = "operationDifferentiateNone"
	operationDifferentiateSub = "operationDifferentiateSub"
	operationDifferentiateAdd = "operationDifferentiateAdd"
)

type operation struct {
	name string
	tensor *Tensor
	children []*operation
	metadata []float64
}

func newOperation(name string, tensor *Tensor, children []*operation, metadata ...float64) *operation {
	return &operation{
		name: name,
		tensor: tensor,
		children: children,
		metadata: metadata,
	}
}

func (o *operation) isLeaf() bool {
	return o.name == operationNone
}
