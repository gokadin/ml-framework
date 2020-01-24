package datasets

type set struct {
	data [][]float64
}

func (s *set) Data() [][]float64 {
	return s.data
}

func (s *set) Normalize(min, max float64) *set {
	normalize(s.data, min, max)

	return s
}

func (s *set) OneHot(depth int) *set {
	oneHotEncode(s.data, depth)

	return s
}
