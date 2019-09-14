package tensor

const (
	creationOperatorAdd = "add"
	creationOperatorNeg = "neg"
	creationOperatorSub = "sub"
	creationOperatorMul = "mul"
	creationOperatorDiv = "div"
	creationOperatorDivScalar = "divScalar"
	creationOperatorPow = "pow"
	creationOperatorTranspose = "transpose"
	creationOperatorDot = "dot"
	creationOperatorSumX = "sumX"
	creationOperatorSumY = "sumY"
	creationOperatorExpandX = "expandX"
	creationOperatorExpandY = "expandY"
	ActivationFunctionSigmoid = "activationFunctionSigmoid"
	ActivationFunctionTanh = "activationFunctionTanh"
	ActivationFunctionIdentity = "activationFunctionIdentity"
)

func (t *Tensor) Backward() {
	t.setInitialGradient()
	t.backpropagate()
}

func (t *Tensor) backpropagate() {
	switch t.creationOperator {
	case creationOperatorMul:
		t.backpropagateMul()
		break
	case creationOperatorDiv:
		t.backpropagateDiv()
		break
	case creationOperatorDivScalar:
		t.backpropagateDivScalar()
		break
	case creationOperatorAdd:
        t.backpropagateAdd()
		break
	case creationOperatorSub:
		t.backpropagateSub()
		break
	case creationOperatorPow:
		t.backpropagatePow()
		break
	case creationOperatorDot:
		t.backpropagateDot()
		break
	case creationOperatorSumX:
		t.backpropagateSumX()
		break
	case creationOperatorExpandX:
		t.backpropagateExpandX()
		break
		// activations
	case ActivationFunctionSigmoid:
        t.backpropagateSigmoid()
		break
	}

	t.continueBackpropagation()
}

func (t *Tensor) continueBackpropagation() {
	for _, creator := range t.creators {
		if !creator.isLeaf() {
			creator.backpropagate()
		}
	}
}

func (t *Tensor) backpropagateAdd() {
	t.creators[0].grad = t.grad
	t.creators[1].grad = t.grad
}

func (t *Tensor) backpropagateSub() {
	t.creators[0].grad = t.grad
	t.creators[1].grad = mulScalar(t.grad, -1)
}

func (t *Tensor) backpropagatePow() {
    t.creators[0].grad = mul(t.grad, mulScalar(pow(t.creators[0].mat, t.creationMetadataFloat64 - 1), t.creationMetadataFloat64))
}

func (t *Tensor) backpropagateMul() {
	t.creators[0].grad = mul(t.grad, t.creators[1].mat)
	t.creators[1].grad = mul(t.grad, t.creators[0].mat)
}

func (t *Tensor) backpropagateDiv() {
	t.creators[0].grad = mul(t.grad, divideScalarBy(t.creators[1].mat, 1))
	t.creators[1].grad = mul(t.grad, divideScalarBy(pow(t.creators[1].mat, 2), 1))
}

func (t *Tensor) backpropagateDivScalar() {
	t.creators[0].grad = mulScalar(t.grad, 1 / t.creationMetadataFloat64)
}

func (t *Tensor) backpropagateDot() {
    t.creators[0].grad = dot(t.grad, transpose(t.creators[1].mat))
    t.creators[1].grad = transpose(dot(transpose(t.grad), t.creators[0].mat))
}

func (t *Tensor) backpropagateSumX() {
	t.creators[0].grad = expand(t.mat, 0, len(t.creators[0].mat))
}

func (t *Tensor) backpropagateExpandX() {
	t.creators[0].grad = sum(t.mat, 0)
}

func (t *Tensor) backpropagateSigmoid() {
	t.creators[0].grad = mul(t.grad, mul(t.mat, subFromScalar(t.mat, 1)))
}

func (t *Tensor) isLeaf() bool {
	return len(t.creators) == 0
}

func (t *Tensor) setInitialGradient() {
	t.grad = make([][]float64, len(t.mat))
	for i := range t.grad {
		t.grad[i] = make([]float64, len(t.mat[i]))
		for j := range t.grad[i] {
			t.grad[i][j] = 1.0
		}
	}
}
