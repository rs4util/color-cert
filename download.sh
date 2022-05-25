mkdir -p ./colorization_model
MODEL_FILE=./colorization_model/pytorch.pth
URL=http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth

wget -N $URL -O $MODEL_FILE