#This script installs the tools for computing bottleneck distance. This has been tested on an Ubuntu deep learning AMI on AWS.
#After building gudhi, set PYTHONPATH to point to the cython module of the gudhi build.

#Install pytorch and other libs for computing embedding similarity and nearest neighbor search.
conda install -y pytorch=0.3.0 -c soumit
pip install torchvision
pip install scipy
pip install nltk

#Download gudhi.
mkdir -p ./tmp
cd ./tmp
wget https://gforge.inria.fr/frs/download.php/file/37362/2018-01-31-09-25-53_GUDHI_2.1.0.tar.gz
tar -xvf 2018-01-31-09-25-53_GUDHI_2.1.0.tar.gz
mv 2018-01-31-09-25-53_GUDHI_2.1.0 gudhi

#install dependencies for CGAL.
sudo apt-get update
sudo apt-get install freeglut3, freeglut3-dev binutils-gold g++ cmake libglew-dev g++ mesa-common-dev build-essential libglew1.5-dev libglm-dev
sudo apt-get install libgmp3-dev
sudo apt-get install libmpfr-dev libmpfr-doc libmpfr4 libmpfr4-dbg

#fix some lib linking issues. Might not be needed depending on the setup.
sudo rm /usr/lib/x86_64-linux-gnu/libGL.so
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1  /usr/lib/x86_64-linux-gnu/libGL.so

#install eigen3.
sudo apt install libeigen3-dev

#install doxygen.
sudo apt-get install doxygen

#make CGAL.
wget https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.8.1/CGAL-4.8.1.tar.xz
tar -xvf CGAL-4.8.1.tar.xz
cd CGAL-4.8.1/
cmake .
make
cd ..

#make gudhi.
cd gudhi
mkdir build
cd build
cmake -DCGAL_DIR=/home/ubuntu/mountdir/CGAL-4.8.1 .. && make
cd ..
