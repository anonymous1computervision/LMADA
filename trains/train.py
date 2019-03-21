from data.dataset import get_data

def train(M, FLAGS):
    print("train called")
    svhn = get_data('svhn', FLAGS)
