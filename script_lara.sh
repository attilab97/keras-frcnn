#comenzile astea doua nu merg apelate din script pt ca nu stie calea catre comanda de conda
#conda create -n BodisAttila python=3.6
#conda activate BodisAttila
mkdir BodisAttila_Lara
cd BodisAttila_Lara
#clonez repo-ul de la mine de pe github
git clone git://github.com/attilab97/keras-frcnn.git

cd ./keras-frcnn
pip install -r requirements.txt
#descarc ponderile
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5


#downloadez datele de la mine de pe drive, in 2 pasi pt ca e un fisier mai mare de 100 mb si atunci in primul pas google imi da o cheie
#folosind cheia de la google descarc efectiv fisierul
#link catre fisier https://drive.google.com/open?id=1ziFRjt8T565-JPofW6Hg0dZ9htgdJTGC
export fileid=1ziFRjt8T565-JPofW6Hg0dZ9htgdJTGC
export filename=Images.zip

## WGET ##
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

#dezarhivez pozele
unzip Images.zip -d ./

rm -f Images.zip

#in folderul Images sunt si pozele si fisierul data.txt unde sunt anotarile
cd ./Images

#antrenez reteaua, ponderile se salveaza automat cand un loss este mai mic decat unul din trecut, se salveaza peste cel de la epoca precedenta
python ../train_frcnn.py  -p data.txt --output_weight_path '../lara_weights.h5' --num_epochs 500 --input_weight_path '../resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
