## Additions to run CMR

- openmesh==1.2.1

### 1. torch_scatter

download **torch_scatter** from https://pytorch-geometric.com/whl/.
in my case, choose [torch-1.11.0+cu113](), and download
[torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl](https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl).

Then, install it in your current `neo` env;

```shell
$ pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
```

Next, use pip install:

```
torch_geometric==2.0.4
torch_sparse==0.6.13
```

the `torch_sparse` package will build for a long time.

### 2. MPI-IS's `mesh` libarary

into the thirdparty folder

```shell
$ git clone https://github.com/MPI-IS/mesh.git
$ cd mesh
```

change the `mesh/requirenment.txt` to match your current `neo` env:

You need to make sure all the below packages' version match your neo' packeges version.
Otherwise, the `make all` process will uninstlled existing neo's packege and reinstall a default version from `pip`.

in my case, I need to fill the `mesh/requirenment.txt` as below:

```
setuptools==58.5.0
numpy==1.21.5
matplotlib==3.5.0
scipy==1.5.2
pyopengl==3.1.6
pillow==9.1.0
pyzmq
pyyaml
opencv-python==4.5.1.48
```

Now, in your `neo` env, execute:

```shell
$ BOOST_INCLUDE_DIRS=/usr/include/boost/ make all
```

Test:
`python -c "from psbody.mesh import Mesh"`

Success !
