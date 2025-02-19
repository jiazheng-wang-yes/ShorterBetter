import numpy as np
def check_efficiency(full_response, accuracy):
    """
    check the efficiency of the model outputs, defined as Acc/avg(len(response))
    """
    if accuracy <= 1:
        accuracy *= 100
    efficiency = accuracy / (np.mean([len(response) for response in full_response])/100)
    return efficiency