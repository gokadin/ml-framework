package data

import "testing"

func TestDataset_GenerateRandomData(t *testing.T) {
	ds := NewDataset()

    ds.FromRandom(10, 1)

	if len(ds.data) != 10 {
		t.Fatal("data size should be 10, got", len(ds.data))
	}
}

func TestDataset_GenerateRandomData_shouldContainExpectedData(t *testing.T) {
	ds := NewDataset()

    ds.FromRandom(10, 1)

	if len(ds.Data()) != 10 {
		t.Fatal("expected set size should be 10, got", len(ds.Data()))
	}
}

func TestDataset_GenerateRandomData_inputSizeShouldBeCorrect(t *testing.T) {
	ds := NewDataset()

    ds.FromRandom(10, 4)

	for _, input := range ds.data {
		if len(input) != 4 {
			t.Fatal("expected input size should be 1, got", len(input))
		}
	}
}

func TestDataset_GenerateRandomData_expectedOutputSizeShouldBeCorrect(t *testing.T) {
	ds := NewDataset()

    ds.FromRandom(10, 1)

	for _, output := range ds.Data() {
		if len(output) != 1 {
			t.Fatal("expected output size should be 1, got", len(output))
		}
	}
}
