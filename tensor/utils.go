package tensor

func handleOpResult(code int) {
	if code == 0 {
		return
	}

	switch code {
	case 1:
		panic("operation failed due to incompatible shapes")
	case -2:
		panic("encountered CUDA error")
	}
}
