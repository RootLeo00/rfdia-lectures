#!/bin/bash
# launch this file with "source launch_tests.sh"
# Explanation: if you launch it with bash or shell, every python invocation will open a 
# new shell that hasn't the conda env activated
# see: https://superuser.com/questions/176783/what-is-the-difference-between-executing-a-bash-script-vs-sourcing-it/176788#176788?newreg=f269a9c96d2f40be9630b907e7913754

conda activate rdfia

echo "Modify ngf or ndf. In particular, reduce or increase one of the two significantly."

test_index=0
python test_gan.py --ngf=64 --savepath="ngf64"
python test_gan.py --ngf=128 --savepath="ngf128"
python test_gan.py --ngf=256 --savepath="ngf256"

python test_gan.py --ndf=64 --savepath="ndf64"
python test_gan.py --ndf=128 --savepath="ndf128"
python test_gan.py --ndf=256 --savepath="ndf256"

echo "Finish $test_index"

test_index=1
echo "Change the learning rate on one or both models."
python test_gan.py --lr-d=0.0001 --lr-g=0.0003 --savepath="lrdg_custom_00001_00003"
python test_gan.py --lr-d=0.1 --lr-g=0.1 --savepath="lrdg_custom_01_01"
python test_gan.py --lr-d=0.1 --savepath="lrd_custom_01"
python test_gan.py --lr-g=0.1 --savepath="lrg_custom_01"

echo "Finish $test_index"

test_index=2
echo "Learn for longer (e.g., 30 epochs) even if it seems that the model already generates correct images."
python test_gan.py --epochs=30 --savepath="epochs30"

echo "Finish $test_index"

test_index=3
echo "Reduce or increase significantly nz (e.g., nz = 10 or 1000)."
python test_gan.py --nz=10 --savepath="nz10"
python test_gan.py --nz=1000 --savepath="nz1000"

echo "Finish $test_index"
