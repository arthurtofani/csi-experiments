git clone https://github.com/MTG/da-tacos.git
cd da-tacos
python3s download_da-tacos.py --dataset metadata --outputdir /dataset --unpack --remove
cd -
rm -rf da-tacos
