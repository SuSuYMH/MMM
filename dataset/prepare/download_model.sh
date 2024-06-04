echo -e "Downloading pretrain"
mkdir -p ./output/vq/
mkdir -p ./output/t2m/
cd ./output
gdown --fuzzy https://drive.google.com/file/d/1Ww0IKHBnloq-Cu854qdfTslGi6o35G0S/view?usp=drive_link
unzip text-to-motion_2localLayer.zip
mv text-to-motion_2localLayer/2023-07-19-04-17-17_12_VQVAE_20batchResetNRandom_8192_32 ./vq/
mv text-to-motion_2localLayer/2023-10-10-03-17-01_HML3D_44_crsAtt2lyr_mask0.5-1 ./t2m/
rm text-to-motion_2localLayer.zip
rm -r text-to-motion_2localLayer
echo -e "Downloading done!"




# echo -e "Downloading pretrain"
# mkdir -p ./output/vq/vq_name/
# cd ./output/vq/vq_name/
# gdown --fuzzy https://drive.google.com/file/d/10ij-p9TR6WVcxVTt5SZSqo22DpVBKGpH/view?usp=drive_link
# mv vqvae.pth net_last.pth
# cd ../../
# mkdir -p ./t2m/trans_name/
# cd ./t2m/trans_name/
# gdown --fuzzy https://drive.google.com/file/d/1EoKM5a-ki5Zs6MqVsjcb1f5KvoUrAmTF/view?usp=drive_link
# mv trans.pth net_last.pth
# echo -e "Downloading done!"