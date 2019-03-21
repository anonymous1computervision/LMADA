class svhn(object):
    def __init__(self, FLAGS):
        print("wow")


def get_data(domain, FLAGS):
    return eval(domain)(FLAGS)
