usage:

python main.py

main control are in params.txt

Main Parameters:
- run is which community method to use (things got messy though)
- build is the network type to use
- community_size can be for global_ and local_community
- num_communities is only for global_community


Caveats: 
- careful to keep params tab-delimited
- only use 1 seed
- changing the seed from string to int requires changing listStr -> listInt
- otherwise datatypes do not need to be changed