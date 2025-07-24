import torch
from traindata_gen.gen_domain_knowledge import gen_domain_knowledge
from traindata_gen.get_structured_data import gen_structured_data
from traindata_gen.make_jonsl import make_jonsl

dataPath = '/mnt/vlr/laishi/IADtraindata'

# -IADtraindata
# --MPDD
# ---bracket_black
# ---IADtraindata/MPDD/bracket_brown
# ---...
# --Real-IAD
# ---audiojack
# ---bottle_cap
# ---...
# --Vision
# ---Cable
# ---Capacitor
# ---...

model_path = '/mnt/vlr/laishi/clip-vit-base-patch32'

k = 10    #number of extracted similar images
batch_size = 256

if __name__ == '__main__':
    gen_structured_data(dataPath, model_path, k=k, batch_size=batch_size,
                        device="cuda" if torch.cuda.is_available() else "cpu")
    gen_domain_knowledge(dataPath)
    make_jonsl(dataPath, number_per_type=100)
