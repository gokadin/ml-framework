package tensor

import "log"

const (
	operationNone = "operationNone"
	operationAdd = "operationAdd"
	operationSub = "operationSub"
)

type operation struct {
	name string
	tensor *Tensor
	gradient [][]float64
	children []*operation
	metadata []float64
	isMarked bool
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

func (o *operation) differentiate() {
    switch o.name {
	case operationNone:
        o.gradient = o.tensor.mat
		break
	case operationAdd:
        o.gradient = findMarkedChild(o).gradient
		break
	case operationSub:
		if o.children[0].isMarked {
			o.gradient = o.children[0].gradient
		} else if o.children[1].isMarked {
			o.gradient = mulScalar(o.children[1].gradient, -1)
		} else {
			log.Fatal("could not find a marked child")
		}
		break
	}
}

func findMarkedChild(operation *operation) *operation {
    for _, child := range operation.children {
    	if child.isMarked {
    		return child
		}
	}
    
    log.Fatalf("could not find a marked child")
    return nil
}
