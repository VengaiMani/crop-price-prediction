import pickle

# with open('map.pkl','rb') as handle:
#     a=pickle.load(handle)
#     print(a)
#     print(type(a))

import configModel as model
model.set(['a','b','c'])
print(model.get("Mannngh",99))