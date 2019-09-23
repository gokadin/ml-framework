package tensor

import (
	"log"
)

const (
	operationNone = "operationNone"
	operationAdd = "operationAdd"
	operationSub = "operationSub"
	operationPow = "operationPow"
	operationDivScalar = "operationDivScalar"
	operationSum = "operationSum"
	operationDot = "operationDot"
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

func (o *operation) differentiate(grad [][]float64) [][]float64 {
    switch o.name {
	case operationNone:
        return o.differentiateNone()
	case operationAdd:
        return o.differentiateAdd()
	case operationSub:
		return o.differentiateSub()
	case operationPow:
		return o.differentiatePow()
	case operationDivScalar:
		return o.differentiateDivScalar()
	case operationSum:
		return o.differentiateSum()
	case operationDot:
		return o.differentiateDot()
	default:
		log.Fatalf("differentiation not supported")
		return nil
	}
}

func (o *operation) differentiateNone() [][]float64 {
	return generateIdentityGradient(o.tensor.mat)
}

func (o *operation) differentiateAdd() [][]float64 {
	return findMarkedChild(o).differentiate()
}

func (o *operation) differentiateSub() [][]float64 {
	if o.children[0].isMarked {
		return o.children[0].differentiate()
	}
	if o.children[1].isMarked {
		return mulScalar(o.children[1].differentiate(), -1)
	}
	log.Fatal("could not find a marked child")
	return nil
}

func (o *operation) differentiatePow() [][]float64 {
	if o.metadata[0] == 2 {
		d := o.children[0].differentiate()
		x := mul(mulScalar(o.children[0].tensor.mat, 2), d)
		return x
	}
	return mul(mulScalar(pow(o.children[0].tensor.mat, o.metadata[0] - 1), o.metadata[0]), o.children[0].differentiate())
}

func (o *operation) differentiateDivScalar() [][]float64 {
	return mulScalar(o.children[0].differentiate(), 1 / o.metadata[0])
}

func (o *operation) differentiateSum() [][]float64 {
    if o.metadata[0] == 0 {
    	return o.differentiateSumX()
	}

    log.Fatal("sum y derivative not yet supported")
	return nil
}

func (o *operation) differentiateSumX() [][]float64 {
	//return mul(expand(o.tensor.mat, 0, len(o.children[0].tensor.mat)), o.children[0].differentiate())
	x := o.children[0].differentiate()
	z := expand(mul(o.tensor.mat, x), 0, len(o.children[0].tensor.mat))
	return z
}

func (o *operation) differentiateDot() [][]float64 {
	if o.children[0].isMarked {
        return mul(dot(o.tensor.mat, transpose(o.children[1].tensor.mat)), o.children[0].differentiate())
	}
	if o.children[1].isMarked {
		x := o.children[1].differentiate()
		a := transpose(o.tensor.mat)
		b := dot(a, o.children[0].tensor.mat)
		c := transpose(b)
		d := mul(c, x)
        return d
		//return mul(transpose(dot(transpose(o.tensor.mat), o.children[0].tensor.mat)), o.children[1].differentiate())
	}
	log.Fatal("could not find a marked child")
	return nil
}

func generateIdentityGradient(mat [][]float64) [][]float64 {
	grad := make([][]float64, len(mat))
	for i := range grad {
		grad[i] = make([]float64, len(mat[i]))
		for j := range grad[i] {
			grad[i][j] = 1
		}
	}
	return grad
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
