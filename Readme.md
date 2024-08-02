# Ring

Repository for the source code of the engine presented in the paper [New Compressed Indices for Multijoins on Graph Databases](https://arxiv.org/abs/2408.00558).

## Instructions

To run our code, **we have to install an extended version of the library SDSL**. Go to [this repository](https://github.com/adriangbrandon/sdsl-lite) and follow the instructions.

After the extended version of SDSL is installed, we have to clone this repository and follow these steps:

1. Create our `build` folder and compile the code:
```Bash
mkdir build
cd build
cmake ..
make
```

Check that there is no errors.

2. Download the version of Wikidata that you want to use: 

- [Wikidata Filtered (about 80M triples)](https://zenodo.org/records/13141588/files/wikidata-filtered-enumerated.tar.gz?download=1).
- [Wikidata (about 1000M triples)](https://zenodo.org/records/13141588/files/wikidata-ring.tar.gz?download=1).

Now put the .dat file inside a folder.

3. Building the index. After compiling the code we should have an executable called `build-index` in `build`. Now run:

```Bash
./build-index <absolute-path-to-the-.dat-file> <type-ring>
```

`<type-ring>` can take four values:
- `ring`, which builds Ring-large
- `c-ring`, which builds Ring-small
- `ring-muthu`, which builds VRing-large
- `c-ring-muthu`, which builds VRing-small
  
This will generate the index in the folder where the `.dat` file is located. The index is suffixed with `.ring`, `.c-ring`, `.ring-muthu`, or `c-ring-muthu` according to the second argument.

In order to build the unidirectional variants use `build-index-uring` instead of `build-index`.

4. Querying the index. In `build` folder, you should find another executable file called `query-index`. To solve the queries you should run:

```Bash
./query-index <absoulute-path-to-the-index-file> <absolute-path-to-the-query-file>
```
There are several variants tha can replace the basic `query-index`:
- Versions which contains `fixed` corresponds with those that use the global VEO. Otherwise, the VEO is adaptive.
- Executables with  `veo-intersection` use the VEO that approximates the intersection by traversing the wavelet tree.
- Variants with `uring` word are for the unidirectional version of the Ring.

Note that the second argument is the path to a file that contains all the queries. The queries of our benchmark are in `Queries`:

- The file `Queries-wikidata-benchmark.txt` can be run with `wikidata-filtered-enumerated.dat`.
- The file `Queries-bgps-limit1000.txt` contains the queries of `wikidata-enumerated.dat`.

After running that command, you should see the number of the query, the number of results, and the elapsed time of each one of the queries with the following format:
```Bash
<query number>;<number of results>;<elapsed time>
```
