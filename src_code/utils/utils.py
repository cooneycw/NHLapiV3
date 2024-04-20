import pickle


def save_data(obj):
    with open('storage/config.pkl', 'wb') as outp:  # Use 'wb' to write in binary mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_data():
    # To load the object back
    with open('storage/config.pkl', 'rb') as inp:  # Use 'rb' to read in binary mode
        obj = pickle.load(inp)
    return obj