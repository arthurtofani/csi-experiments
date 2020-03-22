wget http://www.ime.usp.br/~tofani/covers80.tgz
tar -zxvf covers80.tgz
mv coversongs /dataset
rm covers80.tgz
python3 extract_covers80_features.py
