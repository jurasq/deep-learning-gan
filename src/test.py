import dna

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dna.load_dna_data(100, 400, "../Data", ["Human", "Pig", "Dolphin"], 1)
