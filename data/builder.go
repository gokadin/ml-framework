package data

type builder struct {
	dataset *Dataset
}

func newBuilder(dataset *Dataset) *builder {
	return &builder{
		dataset: dataset,
	}
}

func (b *builder) readCsv(filename string, startIndex, endIndex, limit int) *builder {
	b.dataset.data = readCsv(filename, startIndex, endIndex, limit)
	return b
}

func (b *builder) readRandom(associations, size int) *builder {
    b.dataset.data = readRandom(associations, size)
    return b
}

func (b *builder) normalize(min, max float64, columns ...int) {
	for _, row := range b.dataset.data {
		for _, columnIndex := range columns {
            row[columnIndex] = (row[columnIndex] - min) / (max - min)
		}
	}
}
