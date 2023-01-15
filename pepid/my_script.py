import numpy

if __package__ is None or __package__ == '':
    import blackboard
else:
    from . import blackboard

# This is an example user script containing user functions that may be specified in the config

def predict_length(queries):
    ret = []
    for query in queries:
        ret.append(numpy.random.rand(40).tolist())
    return ret

def length_scorer(cand, query):
   return (cand['mass'] - query['mass'])**2 
    
