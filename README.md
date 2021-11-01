#MacOS/Linux-Ubuntu + Anaconda environments

conda create -n face_recognition python=3.7

conda activate face_recognition

pip install -r requirements.txt

#add photo to data/people_data

python save_face_encoding.py

python checkin.py
