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
	operationExpand = "operationExpand"
	operationDot = "operationDot"
	operationActivationSigmoid = "operationActivationSigmoid"
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

func (o *operation) differentiate(grad [][]float64) {
    switch o.name {
	case operationNone:
        o.differentiateNone(grad)
	case operationAdd:
        o.differentiateAdd(grad)
	case operationSub:
		o.differentiateSub(grad)
	case operationPow:
		o.differentiatePow(grad)
	case operationDivScalar:
		o.differentiateDivScalar(grad)
	case operationSum:
		o.differentiateSum(grad)
	case operationExpand:
		o.differentiateExpand(grad)
	case operationDot:
		o.differentiateDot(grad)
	case operationActivationSigmoid:
		o.differentiateActivationSigmoid(grad)
	default:
		log.Fatalf("differentiation not supported")
	}
}

func (o *operation) differentiateNone(grad [][]float64) [][]float64 {
    return grad
}

func (o *operation) differentiateAdd(grad [][]float64) {
	if o.children[0].tensor.isGradientEnabled {
		o.children[0].gradient = grad
	}
	if o.children[1].tensor.isGradientEnabled {
		o.children[1].gradient = grad
	}
}

func (o *operation) differentiateSub(grad [][]float64) {
	if o.children[0].tensor.isGradientEnabled {
		o.children[0].gradient = grad
	}
	if o.children[1].tensor.isGradientEnabled {
		o.children[1].gradient = mulScalar(grad, -1)
	}
}

func (o *operation) differentiatePow(grad [][]float64) {
	if o.metadata[0] == 2 {
		o.children[0].gradient = mul(grad, mulScalar(o.children[0].tensor.mat, 2))
	}
	o.children[0].gradient = mul(grad, mulScalar(pow(o.children[0].tensor.mat, o.metadata[0] - 1), o.metadata[0]))
}

func (o *operation) differentiateDivScalar(grad [][]float64) {
	o.children[0].gradient = mulScalar(grad, 1 / o.metadata[0])
}

func (o *operation) differentiateSum(grad [][]float64) {
    if o.metadata[0] == 0 {
    	o.differentiateSumX(grad)
    	return
	}

    log.Fatal("sum y derivative not yet supported")
}

func (o *operation) differentiateSumX(grad [][]float64) {
	o.children[0].gradient = expand(grad, 0, len(o.children[0].tensor.mat))
}

func (o *operation) differentiateExpand(grad [][]float64) {
	if o.metadata[0] == 0 {
        o.differentiateExpandX(grad)
        return
	}

	log.Fatal("expand y derivative not yet supported")
}

func (o *operation) differentiateExpandX(grad [][]float64) {
    o.children[0].gradient = sum(grad, 0)
}

func (o *operation) differentiateDot(grad [][]float64) {
	if o.children[0].tensor.isGradientEnabled {
        o.children[0].gradient = dot(grad, transpose(o.children[1].tensor.mat))
	}
	if o.children[1].tensor.isGradientEnabled {
		o.children[1].gradient = transpose(dot(transpose(grad), o.children[0].tensor.mat))
	}
}

func (o *operation) differentiateActivationSigmoid(grad [][]float64) {
	o.children[0].gradient = mul(grad, mul(o.tensor.mat, subFromScalar(o.tensor.mat, 1)))
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
